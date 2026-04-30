import { Injectable, Logger, BadRequestException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { ResolveService } from '../resolve/resolve.service';
import { ProviderKeyService } from '../routing-core/provider-key.service';
import { TierService } from '../routing-core/tier.service';
import { OpenaiOauthService } from '../oauth/openai-oauth.service';
import { MinimaxOauthService } from '../oauth/minimax-oauth.service';
import { ForwardResult } from './provider-client';
import { SessionMomentumService } from './session-momentum.service';
import { LimitCheckService } from '../../notifications/services/limit-check.service';
import { shouldTriggerFallback } from './fallback-status-codes';
import { classifyUpstreamError } from './error-classifier';
import { Tier, ScorerMessage } from '../../scoring/types';
import {
  ProxyFallbackService,
  FailedFallback,
  normalizeProviderModel,
  resolveApiKey,
} from './proxy-fallback.service';
import { ProxyRequestOptions } from './proxy-types';
import { ThoughtSignatureCache } from './thought-signature-cache';
import { buildFriendlyResponse, getDashboardUrl } from './proxy-friendly-response';

export { FailedFallback } from './proxy-fallback.service';

/**
 * Roles excluded from scoring. OpenClaw (and similar tools) inject a large,
 * keyword-rich system prompt with every request. Scoring it inflates every
 * request to the most expensive tier. We strip these before the scorer sees
 * them, but forward the full unmodified body to the real provider.
 */
const SCORING_EXCLUDED_ROLES = new Set(['system', 'developer']);
const SCORING_RECENT_MESSAGES = 10;

export interface RoutingMeta {
  tier: Tier;
  model: string;
  provider: string;
  confidence: number;
  reason: string;
  auth_type?: string;
  fallbackFromModel?: string;
  fallbackIndex?: number;
  primaryErrorStatus?: number;
  primaryErrorBody?: string;
  /**
   * When set, identifies why an automatic escalation/recovery fallback fired
   * (e.g. 'context_overflow_escalation'). Surfaced as `routing_reason` on
   * the recorded agent_message so frequency can be monitored from the API.
   */
  escalationReason?: string;
}

export interface ProxyResult {
  forward: ForwardResult;
  meta: RoutingMeta;
  failedFallbacks?: FailedFallback[];
}

@Injectable()
export class ProxyService {
  private readonly logger = new Logger(ProxyService.name);

  constructor(
    private readonly resolveService: ResolveService,
    private readonly providerKeyService: ProviderKeyService,
    private readonly tierService: TierService,
    private readonly openaiOauth: OpenaiOauthService,
    private readonly minimaxOauth: MinimaxOauthService,
    private readonly momentum: SessionMomentumService,
    private readonly limitCheck: LimitCheckService,
    private readonly fallbackService: ProxyFallbackService,
    private readonly config: ConfigService,
    private readonly signatureCache: ThoughtSignatureCache,
  ) {}

  async proxyRequest(opts: ProxyRequestOptions): Promise<ProxyResult> {
    const { agentId, userId, body, sessionKey, tenantId, agentName, signal } = opts;
    const messages = body.messages;
    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      throw new BadRequestException('messages array is required');
    }
    sanitizeNullContent(messages as Record<string, unknown>[]);

    const limitMessage = await this.enforceLimits(tenantId, agentName);
    if (limitMessage) {
      return buildFriendlyResponse(limitMessage, body.stream === true, 'limit_exceeded');
    }

    const scoringMessages = this.filterScoringMessages(messages as ScorerMessage[]);
    const scoringTools = Array.isArray(body.tools) ? body.tools : undefined;
    const isHeartbeat = this.detectHeartbeat(scoringMessages);
    const recentTiers = this.momentum.getRecentTiers(sessionKey);

    const resolved = isHeartbeat
      ? await this.resolveService.resolveForTier(agentId, 'simple')
      : await this.resolveService.resolve(
          agentId,
          scoringMessages,
          scoringTools,
          body.tool_choice,
          body.max_tokens as number | undefined,
          recentTiers,
        );

    if (!resolved.model || !resolved.provider) {
      this.logger.warn(
        `No model available for agent=${agentId}: ` +
          `tier=${resolved.tier} model=${resolved.model} provider=${resolved.provider} ` +
          `confidence=${resolved.confidence} reason=${resolved.reason}`,
      );
      return this.buildNoProviderResult(body.stream === true, agentName);
    }

    let apiKey = await this.providerKeyService.getProviderApiKey(
      agentId,
      resolved.provider,
      resolved.auth_type,
    );
    if (apiKey === null) {
      const dashboardUrl = getDashboardUrl(this.config, agentName);
      const content = `[🦚 Manifest] No API key set for ${resolved.provider} yet. Add one here: ${dashboardUrl}`;
      return buildFriendlyResponse(content, body.stream === true, 'no_provider_key');
    }

    const resolvedCredentials = await resolveApiKey(
      resolved.provider,
      apiKey,
      resolved.auth_type,
      agentId,
      userId,
      this.openaiOauth,
      this.minimaxOauth,
    );
    const providerRegion = await this.providerKeyService.getProviderRegion(
      agentId,
      resolved.provider,
      resolved.auth_type,
    );
    const primaryModel = normalizeProviderModel(resolved.provider, resolved.model);

    this.logger.log(
      `Proxy: tier=${resolved.tier} model=${primaryModel} provider=${resolved.provider} auth_type=${resolved.auth_type} confidence=${resolved.confidence}`,
    );

    const stream = body.stream === true;
    const signatureLookup = (toolCallId: string) =>
      this.signatureCache.retrieve(sessionKey, toolCallId);
    const forward = await this.fallbackService.tryForwardToProvider({
      provider: resolved.provider,
      apiKey: resolvedCredentials.apiKey,
      model: primaryModel,
      body,
      stream,
      sessionKey,
      signal,
      authType: resolved.auth_type,
      resourceUrl: resolvedCredentials.resourceUrl,
      providerRegion,
      signatureLookup,
    });

    if (!forward.response.ok && shouldTriggerFallback(forward.response.status)) {
      const tiers = await this.tierService.getTiers(agentId);
      const assignment = tiers.find((t) => t.tier === resolved.tier);
      const tierFallbackModels = assignment?.fallback_models ?? [];

      // Reading the body consumes the original Response stream. We need it
      // for both classification (escalation decision) and the fall-through
      // return path (so downstream can re-emit the original error). Rebuild
      // a fresh Response on `forward` so it stays readable below.
      const primaryStatus = forward.response.status;
      const primaryErrorBody = await forward.response.text();
      const rebuiltHeaders = new Headers(forward.response.headers);
      rebuiltHeaders.delete('content-encoding');
      rebuiltHeaders.delete('content-length');
      rebuiltHeaders.delete('transfer-encoding');
      forward.response = new Response(primaryErrorBody, {
        status: primaryStatus,
        headers: rebuiltHeaders,
      });

      // Context-overflow escalation: when the upstream rejected the request
      // because the prompt exceeded the model's context window, prepend the
      // agent's reasoning-tier model to the fallback chain. The reasoning
      // tier typically maps to a model with a much larger context window
      // (e.g. Sonnet 4.6's 1M tokens vs Haiku 4.5's 200k), so the escalated
      // call can succeed where the original could not. Skipped if we're
      // already on the reasoning tier (escalating to ourselves is a no-op).
      let escalationEntry: { model: string; provider: string } | null = null;
      if (
        resolved.tier !== 'reasoning' &&
        classifyUpstreamError(primaryStatus, primaryErrorBody) === 'context_overflow'
      ) {
        const reasoning = await this.resolveService.resolveForTier(agentId, 'reasoning');
        if (reasoning?.model && reasoning.provider) {
          escalationEntry = { model: reasoning.model, provider: reasoning.provider };
          this.logger.log(
            `Context overflow on tier=${resolved.tier} model=${primaryModel} — escalating to reasoning-tier model=${reasoning.model} provider=${reasoning.provider}`,
          );
        }
      }

      const fallbackModels: Array<string | { model: string; provider: string }> = escalationEntry
        ? [escalationEntry, ...tierFallbackModels]
        : [...tierFallbackModels];

      if (fallbackModels.length > 0) {
        const { success, failures } = await this.fallbackService.tryFallbacks(
          agentId,
          userId,
          fallbackModels,
          body,
          stream,
          sessionKey,
          primaryModel,
          signal,
          resolved.provider ?? undefined,
          resolved.auth_type,
          signatureLookup,
        );

        if (success) {
          // The escalation hop is always at fallbackIndex 0 (we prepended it)
          const isEscalationHit = escalationEntry !== null && success.fallbackIndex === 0;
          this.momentum.recordTier(
            sessionKey,
            (isEscalationHit ? 'reasoning' : resolved.tier) as Tier,
          );
          return {
            forward: success.forward,
            meta: {
              tier: (isEscalationHit ? 'reasoning' : resolved.tier) as Tier,
              model: success.model,
              provider: success.provider,
              confidence: resolved.confidence,
              reason: resolved.reason,
              auth_type: resolved.auth_type,
              fallbackFromModel: primaryModel,
              fallbackIndex: success.fallbackIndex,
              primaryErrorStatus: primaryStatus,
              primaryErrorBody: primaryErrorBody,
              escalationReason: isEscalationHit ? 'context_overflow_escalation' : undefined,
            },
            failedFallbacks: failures,
          };
        }

        // All fallbacks exhausted — preserve the primary provider's real
        // HTTP status. The gateway uses the X-Manifest-Fallback-Exhausted
        // header (set by the response handler) to detect this case.
        const safeHeaders = new Headers(forward.response.headers);
        safeHeaders.delete('content-encoding');
        safeHeaders.delete('content-length');
        safeHeaders.delete('transfer-encoding');

        const rebuilt = new Response(primaryErrorBody, {
          status: primaryStatus,
          headers: safeHeaders,
        });
        this.momentum.recordTier(sessionKey, resolved.tier as Tier);
        return {
          forward: {
            response: rebuilt,
            isGoogle: forward.isGoogle,
            isAnthropic: forward.isAnthropic,
            isChatGpt: forward.isChatGpt,
          },
          meta: {
            tier: resolved.tier as Tier,
            model: primaryModel,
            provider: resolved.provider,
            confidence: resolved.confidence,
            reason: resolved.reason,
            auth_type: resolved.auth_type,
          },
          failedFallbacks: failures,
        };
      }
    }

    this.momentum.recordTier(sessionKey, resolved.tier as Tier);

    return {
      forward,
      meta: {
        tier: resolved.tier as Tier,
        model: primaryModel,
        provider: resolved.provider,
        confidence: resolved.confidence,
        reason: resolved.reason,
        auth_type: resolved.auth_type,
      },
    };
  }

  private async enforceLimits(tenantId?: string, agentName?: string): Promise<string | null> {
    if (!tenantId || !agentName) return null;
    const exceeded = await this.limitCheck.checkLimits(tenantId, agentName);
    if (!exceeded) return null;

    const fmt =
      exceeded.metricType === 'cost'
        ? `$${Number(exceeded.actual).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
        : Number(exceeded.actual).toLocaleString(undefined, { maximumFractionDigits: 0 });
    const threshFmt =
      exceeded.metricType === 'cost'
        ? `$${Number(exceeded.threshold).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
        : Number(exceeded.threshold).toLocaleString(undefined, { maximumFractionDigits: 0 });
    const dashboardUrl = getDashboardUrl(this.config, agentName);
    return `[🦚 Manifest] Usage limit hit: ${exceeded.metricType} is at ${fmt} (limit: ${threshFmt}/${exceeded.period}). You can adjust it here: ${dashboardUrl}`;
  }

  private filterScoringMessages(messages: ScorerMessage[]): ScorerMessage[] {
    return messages
      .filter((m) => !SCORING_EXCLUDED_ROLES.has(m.role))
      .slice(-SCORING_RECENT_MESSAGES);
  }

  private detectHeartbeat(scoringMessages: ScorerMessage[]): boolean {
    const lastUser = [...scoringMessages].reverse().find((m) => m.role === 'user');
    if (!lastUser) return false;
    if (typeof lastUser.content === 'string') return lastUser.content.includes('HEARTBEAT_OK');
    if (Array.isArray(lastUser.content)) {
      return (lastUser.content as { type?: string; text?: string }[]).some(
        (p) => p.type === 'text' && typeof p.text === 'string' && p.text.includes('HEARTBEAT_OK'),
      );
    }
    return false;
  }

  private buildNoProviderResult(stream: boolean, agentName?: string): ProxyResult {
    const dashboardUrl = getDashboardUrl(this.config, agentName);
    const content = `[🦚 Manifest] Manifest is connected successfully. To start routing requests, connect a model provider: ${dashboardUrl}`;
    return buildFriendlyResponse(content, stream, 'no_provider');
  }
}

/** Replace null content fields with empty string to avoid upstream rejections. */
function sanitizeNullContent(messages: Record<string, unknown>[]): void {
  for (const msg of messages) {
    if (msg && typeof msg === 'object' && msg.content === null) msg.content = '';
  }
}
