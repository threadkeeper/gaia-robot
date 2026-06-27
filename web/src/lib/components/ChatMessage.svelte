<script lang="ts">
  import { debug } from '$lib/stores/settings';
  import type { ChatMessage, RouterRound } from '$lib/types';

  let { message }: { message: ChatMessage } = $props();

  const isUser = $derived(message.role === 'user');

  const SEARCH_LABELS: Record<string, string> = {
    T1_THINK: 'T1 think',
    T2_THINK: 'T2 think',
    T3_THINK: 'T3 think',
    SEARCH_LTM: 'Search LTM',
    SEARCH_KB: 'Search KB',
    SEARCH_WEB: 'Search Web'
  };
  const ROUTER_SKILLS = ['T1_THINK', 'T2_THINK', 'T3_THINK', 'SEARCH_LTM', 'SEARCH_KB', 'SEARCH_WEB'];
  const ROUTER_ACTIONS = ['SEND_MESSAGE', 'SEND_PUSH', 'ADD_STM', 'ADD_LTM', 'ADD_KB'];

  function searchLabel(kind: string): string {
    return SEARCH_LABELS[kind] ?? kind;
  }

  /**
   * Map an operation / skill / action identifier to a debug-tile colour class.
   * Colour key: T1 light blue, T2 blue, T3 dark blue, search STM light yellow,
   * search KB yellow, search LTM gold, search web green, add STM pink,
   * add KB red, add LTM indigo, send message violet, routing grey,
   * encrypt black, everything else white.
   */
  function tileCategory(raw: string): string {
    const s = raw.toLowerCase();
    if (
      s.includes('self_model') ||
      s.includes('manifested') ||
      s.startsWith('gaia.') ||
      s.includes('self_research') ||
      s.includes('self_build') ||
      s.includes('diary')
    )
      return 'cat-gaia-think';
    if (s.includes('encrypt') || s.includes('merkle') || s.includes('seal')) return 'cat-encrypt';
    if (s.includes('assess_emotion') || s.includes('emotion')) return 'cat-emotion';
    if (s.includes('friendship')) return 'cat-friendship';
    if (s.startsWith('router') || s.includes('round_start') || s.includes('round_done')) return 'cat-routing';
    if (
      s.includes('send_message') ||
      s.includes('send_push') ||
      s.includes('stream_to_client') ||
      s.includes('push_notification') ||
      s.includes('notify')
    )
      return 'cat-send';
    if (s.includes('add_stm') || s.includes('append_user') || s.includes('append_gaia')) return 'cat-add-stm';
    if (s.includes('add_kb') || s.includes('kg_upsert') || s.includes('kg_add')) return 'cat-add-kb';
    if (s.endsWith('.manifest') || s === 'manifest') return 'cat-manifest';
    if (s.includes('add_ltm') || s.includes('ltm.write') || s.includes('ltm_write') || s.includes('write_turn'))
      return 'cat-add-ltm';
    if (s.includes('search_web') || s.includes('web_search')) return 'cat-search-web';
    if (s.includes('search_ltm') || s.includes('ltm_recall')) return 'cat-search-ltm';
    if (s.includes('search_kb') || s.includes('kg.summary') || s.includes('kg_query') || s.includes('knowledge'))
      return 'cat-search-kb';
    if (s.includes('search_stm') || s.includes('set_recall') || s.includes('stm.read') || s.includes('diary_read'))
      return 'cat-search-stm';
    if (s.includes('t3')) return 'cat-t3';
    if (s.includes('t2')) return 'cat-t2';
    if (s.includes('t1')) return 'cat-t1';
    return 'cat-other';
  }

  function plannerLine(round: RouterRound): string {
    const planned = round.plannedSkills ?? [];
    if (!planned.length) return '';
    return planned.map(searchLabel).join(' • ');
  }

  function evidenceFor(round: RouterRound, skill: string): string {
    const hit = (round.skillEvidence ?? []).find((e) => e.skill === skill);
    if (!hit) return '';
    if (hit.error) return `error: ${hit.error}`;
    const kb = (hit.kb ?? '').trim();
    if (kb) return kb;
    if (hit.resultCount === 0) return '(no results)';
    return '';
  }

  function actionFor(action: string): { action: string; status: string; detail?: string } | null {
    const list = message.meta?.actions ?? [];
    return list.find((a) => a.action === action) ?? null;
  }
</script>

<div class="row" class:user={isUser}>
  <div class="bubble" class:user={isUser} class:err={!!message.error}>
    {#if message.error}
      <span class="error-text">⚠ {message.error}</span>
    {:else}
      <span class="text">{message.text}</span>{#if message.streaming}<span class="caret"></span>{/if}
    {/if}

    {#if message.meta && !message.error}
      {@const searches = message.meta.searches ?? []}
      {#if searches.length}
        <div class="skills" title="Skills Gaia used this turn">
          {#each searches as search}
            <span class="chip search">{searchLabel(search)}</span>
          {/each}
        </div>
      {/if}
      {#if $debug}
        <div class="meta">
          <span class="chip" title="Reasoning tier">{message.meta.routing}</span>
          <span class="chip" title="Safety verdict">{message.meta.verdict}</span>
          {#if message.meta.emotions}
            {@const emo = message.meta.emotions}
            <span class="chip emo" title="User emotion: happiness">happiness {emo.happiness.toFixed(2)}</span>
            <span class="chip emo" title="User emotion: sadness">sadness {emo.sadness.toFixed(2)}</span>
            <span class="chip emo" title="User emotion: fear">fear {emo.fear.toFixed(2)}</span>
            <span class="chip emo" title="User emotion: disgust">disgust {emo.disgust.toFixed(2)}</span>
            <span class="chip emo" title="User emotion: anger">anger {emo.anger.toFixed(2)}</span>
            <span class="chip emo" title="User emotion: surprise">surprise {emo.surprise.toFixed(2)}</span>
          {/if}
          <span class="chip" title="Attention weight (average of emotions)">att {message.meta.attention.toFixed(2)}</span>
        </div>
        {@const rounds = message.meta.routerRounds ?? []}
        {@const toolDebug = message.meta.debug ?? []}
        {@const actions = message.meta.actions ?? []}
        {@const flow = (message.meta.flow ?? []).slice().sort((a, b) => a.seq - b.seq)}
        {@const pullDebug = message.meta.pullDebug}
        {@const pushDebug = message.meta.pushDebug}
        {#if pullDebug || pushDebug}
          <details class="debug" open>
            <summary>pass debug (pull/push model · llm time · actions)</summary>

            {#if pullDebug}
              <div class="round">
                <div class="round-head">
                  <strong>Pull pass (LLM Call 1)</strong>
                  <span class="ready">{pullDebug.llmMs} ms</span>
                </div>
                <div class="skills-grid">
                  <span class="chip skill used" title="Model selected for the pull pass">
                    <span>model</span>
                    <span class="kb">{pullDebug.model}</span>
                  </span>
                  <span class="chip skill used" title="Time spent in the LLM Call 1 request">
                    <span>llm time</span>
                    <span class="kb">{pullDebug.llmMs} ms</span>
                  </span>
                </div>
                {#if pullDebug.actions.length}
                  <div class="skills-grid">
                    {#each pullDebug.actions as act}
                      <span class="chip skill used" title="Retrieval action chosen by Call 1 and the time it took">
                        <span>{act.type}</span>
                        <span class="kb">{act.ms.toFixed(2)} ms</span>
                      </span>
                    {/each}
                  </div>
                {:else}
                  <div class="planner">No retrieval actions chosen.</div>
                {/if}
              </div>
            {/if}

            {#if pushDebug}
              <div class="round">
                <div class="round-head">
                  <strong>Push pass (LLM Call 2)</strong>
                  <span class="ready">{pushDebug.llmMs} ms</span>
                </div>
                <div class="skills-grid">
                  <span class="chip skill used" title="Model selected for the push pass">
                    <span>model</span>
                    <span class="kb">{pushDebug.model}</span>
                  </span>
                  <span class="chip skill used" title="Time spent in the LLM Call 2 request">
                    <span>llm time</span>
                    <span class="kb">{pushDebug.llmMs} ms</span>
                  </span>
                </div>
                {#if pushDebug.actions.length}
                  <div class="skills-grid">
                    {#each pushDebug.actions as act}
                      <span class="chip skill used" title="Planned action type and time to process">
                        <span>{act.type}</span>
                        <span class="kb">{act.ms.toFixed(2)} ms</span>
                      </span>
                    {/each}
                  </div>
                {:else}
                  <div class="planner">No actions planned.</div>
                {/if}
              </div>
            {/if}
          </details>
        {/if}
        {#if rounds.length || toolDebug.length || actions.length || flow.length}
          <details class="debug" open>
            <summary>router debug (rounds: {rounds.length}, events: {toolDebug.length})</summary>

            {#if flow.length}
              <div class="round">
                <div class="round-head">
                  <strong>Operation Sequence</strong>
                  <span class="ready">steps {flow.length}</span>
                </div>
                <div class="skills-grid">
                  {#each flow as op}
                    <span
                      class="chip skill used {tileCategory(op.op)}"
                      class:checkered={op.source === 'self_research'}
                      title={op.detail || op.op}
                    >
                      <span>#{op.seq} {op.op}</span>
                      <span class="kb">{op.status}</span>
                      {#if op.tokens !== undefined}
                        <span class="kb" title="Context window fed to this tier">{op.tokens}tk</span>
                      {/if}
                    </span>
                  {/each}
                </div>
              </div>
            {/if}

            {#if actions.length}
              <div class="round">
                <div class="round-head">
                  <strong>Action Phase</strong>
                </div>
                <div class="skills-grid">
                  {#each ROUTER_ACTIONS as actionName}
                    {@const hit = actionFor(actionName)}
                    <span
                      class="chip skill {tileCategory(actionName)}"
                      class:used={!!hit}
                      title={hit?.detail || actionName}
                    >
                      <span>{actionName}</span>
                      {#if hit}
                        <span class="kb">{hit.status}</span>
                      {/if}
                    </span>
                  {/each}
                </div>
              </div>
            {/if}

            {#each rounds as rr}
              <div class="round">
                <div class="round-head">
                  <strong>Router Round {rr.round}</strong>
                  <span class="ready">Ready ({rr.ready ? 'yes' : 'no'})</span>
                </div>
                {#if plannerLine(rr)}
                  <div class="planner">Planner: {plannerLine(rr)}</div>
                {/if}
                <div class="skills-grid">
                  {#each ROUTER_SKILLS as skill}
                    {@const used = rr.skillsUsed.includes(skill)}
                    {@const kb = used ? evidenceFor(rr, skill) : ''}
                    <span
                      class="chip skill {tileCategory(skill)}"
                      class:used={used}
                      title={kb || searchLabel(skill)}
                    >
                      <span>{searchLabel(skill)}</span>
                      {#if kb}
                        <span class="kb">{kb}</span>
                      {/if}
                    </span>
                  {/each}
                </div>
              </div>
            {/each}

            {#each toolDebug as entry}
              <div class="dbg-tool">
                <div class="dbg-head">
                  <span class="chip">{entry.tool}</span>
                  {#if entry.round !== undefined}
                    <span class="chip">round {entry.round}</span>
                  {/if}
                  {#if entry.gated === false}
                    <span class="chip warn">skipped · T1 said no search needed</span>
                  {/if}
                  {#if entry.backend}
                    <span class="chip">backend: {entry.backend}</span>
                  {/if}
                  {#if entry.resultCount !== undefined}
                    <span class="chip">{entry.resultCount} result{entry.resultCount === 1 ? '' : 's'}</span>
                  {/if}
                </div>
                {#if entry.backendTrace && entry.backendTrace.length}
                  <div class="dbg-trace">
                    {#each entry.backendTrace as line}
                      <span class="chip muted">{line}</span>
                    {/each}
                  </div>
                {/if}
                {#if entry.warning}
                  <div class="dbg-warn">⚠ {entry.warning}</div>
                {/if}
                {#if entry.error}
                  <div class="dbg-err">⚠ {entry.error}</div>
                {/if}
              </div>
            {/each}
          </details>
        {/if}
      {/if}
    {/if}
  </div>
</div>

{#if !isUser && !message.error && message.meta?.actionsSummary}
  <!-- A second, lighter bubble summarizing the side effects Gaia performed
       this turn (WhatsApp / Push / Edwino actuate / data-store write-backs). -->
  <div class="row">
    <div class="bubble actions" title="Actions Gaia performed this run">
      <span class="actions-head">Actions this run</span>
      <span class="actions-body">{message.meta.actionsSummary}</span>
    </div>
  </div>
{/if}

{#if !isUser && !message.error && message.meta?.write}
  <!-- Always-visible (not debug-gated) banner for the mandatory Cosmos
       write-back. A failure here is shown prominently in red so the user
       knows persistence is broken, even though the answer itself is valid. -->
  <div class="row">
    <div
      class="bubble write"
      class:write-err={!message.meta.write.ok}
      title="Cosmos write-back status for this turn"
    >
      <span class="write-head">
        {message.meta.write.ok ? '✓ Saved to Cosmos' : '⚠ Cosmos write failed'}
      </span>
      <span class="write-body">{message.meta.write.detail}</span>
    </div>
  </div>
{/if}

<style>
  .row {
    display: flex;
    margin: 6px 0;
  }
  .row.user {
    justify-content: flex-end;
  }
  .bubble {
    max-width: min(80ch, 78%);
    padding: 10px 14px;
    border-radius: var(--radius);
    background: var(--gaia-bubble);
    border: 1px solid var(--border);
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
    box-shadow: var(--shadow);
  }
  .bubble.user {
    background: var(--user-bubble);
    border-color: transparent;
  }
  .bubble.err {
    border-color: var(--danger);
  }
  .bubble.actions {
    background: transparent;
    border-style: dashed;
    box-shadow: none;
    font-size: 0.85em;
    opacity: 0.9;
  }
  .actions-head {
    display: block;
    font-weight: 600;
    margin-bottom: 2px;
    opacity: 0.7;
  }
  .actions-body {
    white-space: pre-wrap;
  }
  /* Mandatory write-back banner. Success is a quiet confirmation; failure is
     loud (red border + tint) per the requirement that write problems be
     visible to the user rather than silently swallowed. */
  .bubble.write {
    background: transparent;
    border-style: dashed;
    box-shadow: none;
    font-size: 0.85em;
    opacity: 0.9;
  }
  .bubble.write.write-err {
    border-color: var(--danger);
    border-style: solid;
    background: color-mix(in srgb, var(--danger) 12%, transparent);
    opacity: 1;
  }
  .write-head {
    display: block;
    font-weight: 600;
    margin-bottom: 2px;
    opacity: 0.7;
  }
  .bubble.write.write-err .write-head {
    color: var(--danger);
    opacity: 1;
  }
  .write-body {
    white-space: pre-wrap;
  }
  .error-text {
    color: var(--danger);
  }
  .caret {
    display: inline-block;
    width: 8px;
    height: 1.05em;
    margin-left: 2px;
    vertical-align: text-bottom;
    background: var(--accent);
    border-radius: 2px;
    animation: blink 1s steps(2, start) infinite;
  }
  @keyframes blink {
    50% {
      opacity: 0;
    }
  }
  .meta {
    display: flex;
    gap: 6px;
    margin-top: 8px;
    flex-wrap: wrap;
  }
  .skills {
    display: flex;
    gap: 6px;
    margin-top: 8px;
    flex-wrap: wrap;
  }
  .chip {
    font-size: 11px;
    color: var(--text-dim);
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 2px 8px;
    font-family: var(--mono);
  }
  .chip.search {
    color: var(--accent-2);
  }
  .chip.emo {
    color: var(--accent);
    border-color: var(--accent);
  }
  .chip.warn {
    color: var(--danger);
    border-color: var(--danger);
  }
  .chip.skill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: #8c96a8;
    border-color: #596275;
  }
  .chip.skill.used {
    color: #0a2617;
    background: #8df3bf;
    border-color: #56d892;
  }
  /* Debug-tile colour coding by operation category. Applied to active tiles.
     The cat-* / checkered classes are assigned dynamically, so they are wrapped
     in :global(...) to stop Svelte's compiler from pruning them as unused. */
  .chip.skill.used:global(.cat-t1) {
    color: #0a2a4a;
    background-color: #cfe8ff;
    border-color: #9fcdf5;
  }
  .chip.skill.used:global(.cat-t2) {
    color: #ffffff;
    background-color: #4a90e2;
    border-color: #2f6fbf;
  }
  .chip.skill.used:global(.cat-t3) {
    color: #e8f0ff;
    background-color: #173a6b;
    border-color: #0e2347;
  }
  .chip.skill.used:global(.cat-search-stm) {
    color: #4a3f00;
    background-color: #fff6c2;
    border-color: #f0e08a;
  }
  .chip.skill.used:global(.cat-search-kb) {
    color: #3a3300;
    background-color: #ffe34d;
    border-color: #e6c200;
  }
  .chip.skill.used:global(.cat-search-ltm) {
    color: #3a2e00;
    background-color: #f5c518;
    border-color: #d4a800;
  }
  .chip.skill.used:global(.cat-search-web) {
    color: #06210a;
    background-color: #4caf50;
    border-color: #3a8f3e;
  }
  .chip.skill.used:global(.cat-add-stm) {
    color: #4a0024;
    background-color: #ff9ec4;
    border-color: #f06fa3;
  }
  .chip.skill.used:global(.cat-add-kb) {
    color: #ffffff;
    background-color: #e53935;
    border-color: #b71c1c;
  }
  .chip.skill.used:global(.cat-add-ltm) {
    color: #ffffff;
    background-color: #5c4ddb;
    border-color: #3f33b0;
  }
  .chip.skill.used:global(.cat-manifest) {
    color: #04201d;
    background-color: #34e0c8;
    border-color: #13b9a2;
  }
  .chip.skill.used:global(.cat-send) {
    color: #ffffff;
    background-color: #9b59d0;
    border-color: #7a3fb0;
  }
  .chip.skill.used:global(.cat-routing) {
    color: #1c1f24;
    background-color: #9aa0aa;
    border-color: #7a818c;
  }
  .chip.skill.used:global(.cat-encrypt) {
    color: #f0f0f5;
    background-color: #1b1b1f;
    border-color: #3a3a42;
  }
  .chip.skill.used:global(.cat-friendship) {
    color: #ffffff;
    background-color: #800020;
    border-color: #5c0017;
  }
  .chip.skill.used:global(.cat-emotion) {
    color: #ffffff;
    background-color: #6a3fb0;
    background-image: linear-gradient(
      90deg,
      #e53935 0%,
      #fb8c00 20%,
      #fdd835 40%,
      #43a047 60%,
      #1e88e5 80%,
      #8e24aa 100%
    );
    border-color: #4a4a52;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.55);
  }
  .chip.skill.used:global(.cat-other) {
    color: #1c1f24;
    background-color: #ffffff;
    border-color: #d0d4da;
  }
  /* Gaia-think: Gaia's own thinking / self-build tiles — gold with a sparkle. */
  .chip.skill.used:global(.cat-gaia-think) {
    position: relative;
    overflow: hidden;
    color: #4a3500;
    background-color: #ffd24d;
    background-image: linear-gradient(135deg, #ffe9a8 0%, #ffd24d 50%, #f4b400 100%);
    border-color: #e0a400;
    box-shadow: 0 0 8px rgba(255, 200, 60, 0.55);
    text-shadow: 0 1px 1px rgba(255, 255, 255, 0.45);
  }
  /* Keep tile content above the animated shine sweep. */
  .chip.skill.used:global(.cat-gaia-think) > span {
    position: relative;
    z-index: 1;
  }
  /* Sparkle: a light beam sweeps across the gold tile. */
  .chip.skill.used:global(.cat-gaia-think)::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(
      115deg,
      transparent 30%,
      rgba(255, 255, 255, 0.85) 50%,
      transparent 70%
    );
    transform: translateX(-120%);
    animation: gaia-sparkle 2.6s ease-in-out infinite;
    pointer-events: none;
  }
  @keyframes gaia-sparkle {
    0% {
      transform: translateX(-120%);
    }
    60%,
    100% {
      transform: translateX(120%);
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .chip.skill.used:global(.cat-gaia-think)::before {
      animation: none;
    }
  }
  /* Checkered overlay marks operations on the Gaia self wing (vs. user wing). */
  .chip.skill.used:global(.checkered) {
    background-image: linear-gradient(
        45deg,
        rgba(0, 0, 0, 0.16) 25%,
        transparent 25%,
        transparent 75%,
        rgba(0, 0, 0, 0.16) 75%
      ),
      linear-gradient(
        45deg,
        rgba(0, 0, 0, 0.16) 25%,
        transparent 25%,
        transparent 75%,
        rgba(0, 0, 0, 0.16) 75%
      );
    background-size: 8px 8px;
    background-position:
      0 0,
      4px 4px;
  }
  .kb {
    max-width: 32ch;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: inherit;
    opacity: 0.88;
  }
  .debug {
    margin-top: 8px;
    font-size: 12px;
    font-family: var(--mono);
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 6px 10px;
  }
  .debug summary {
    cursor: pointer;
    color: var(--text-dim);
    user-select: none;
  }
  .dbg-tool {
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid var(--border);
  }
  .round {
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid var(--border);
  }
  .round-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }
  .planner {
    margin-top: 6px;
    color: var(--text-dim);
  }
  .ready {
    color: var(--text-dim);
  }
  .skills-grid {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin-top: 8px;
  }
  .dbg-head {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    align-items: center;
  }
  .dbg-err {
    color: var(--danger);
    margin-top: 6px;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .dbg-warn {
    color: var(--warn, #d08a00);
    margin-top: 6px;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .dbg-trace {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin-top: 6px;
  }
  .chip.muted {
    color: var(--text-dim);
    border-color: var(--border);
    opacity: 0.85;
  }
</style>
