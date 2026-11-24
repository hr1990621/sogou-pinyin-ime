/* Sogou-Style Chinese Pinyin IME Frontend Controller */

(() => {
  const PAGE_SIZE = 5;

  // Chinese punctuation mapping (Sogou default full-width + smart quote pairing)
  const CN_PUNCT = {
    ",": "，",
    ".": "。",
    "?": "？",
    "!": "！",
    ":": "：",
    ";": "；",
    "(": "（",
    ")": "）",
    "[": "【",
    "]": "】",
    "<": "《",
    ">": "》",
    "\\": "、",
    "^": "……",
    "_": "——",
  };

  const state = {
    chineseMode: true,
    cnPunct: true,
    enableAuto: true,
    enableMem: true,
    quoteDoubleOpen: true,
    quoteSingleOpen: true,

    // Active composition buffer
    rawPinyin: "",
    displayPinyin: "",
    pendingWords: [],   // progressive selections within one composition session: [{word, pinyin}]
    candidates: [],
    page: 0,
    activeIndex: 0,     // index within current page (0..PAGE_SIZE-1)

    // Post-commit Association & Inline Ghost Text
    assocMode: false,
    associations: [],
    ghostText: "",
    ghostSource: "",

    // Shift tap detection (tap Shift alone -> toggle CN/EN)
    shiftDownAlone: false,
    querySeq: 0,
    events: [],
  };

  // DOM references
  const $ = (id) => document.getElementById(id);
  const editor = $("editorInput");
  const mirror = $("editorMirror");
  const editorStage = $("editorStage");
  const imePopup = $("imePopup");
  const imeAssocBadge = $("imeAssocBadge");
  const imePendingCn = $("imePendingCn");
  const imePinyinText = $("imePinyinText");
  const imeGhostHint = $("imeGhostHint");
  const imeLatency = $("imeLatency");
  const imeCandidates = $("imeCandidates");
  const pagerInfo = $("pagerInfo");
  const btnPrevPage = $("btnPrevPage");
  const btnNextPage = $("btnNextPage");
  const ghostStatusText = $("ghostStatusText");
  const charCount = $("charCount");

  const btnToggleLang = $("btnToggleLang");
  const btnTogglePunct = $("btnTogglePunct");
  const btnToggleAuto = $("btnToggleAuto");
  const btnToggleMem = $("btnToggleMem");

  // ---------- API Helpers ----------
  async function apiGet(url) {
    const res = await fetch(url);
    return res.json();
  }

  async function apiPost(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return res.json();
  }

  async function apiDelete(url, body) {
    const res = await fetch(url, {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return res.json();
  }

  // ---------- Editor Helpers ----------
  function getContextBeforeCaret() {
    const pos = editor.selectionStart ?? editor.value.length;
    const before = editor.value.slice(0, pos);
    const pendingStr = state.pendingWords.map((p) => p.word).join("");
    return (before + pendingStr).slice(-32);
  }

  function insertTextAtCaret(text) {
    const start = editor.selectionStart ?? editor.value.length;
    const end = editor.selectionEnd ?? editor.value.length;
    const val = editor.value;
    editor.value = val.slice(0, start) + text + val.slice(end);
    const newPos = start + text.length;
    editor.selectionStart = editor.selectionEnd = newPos;
    updateCharCount();
    renderMirrorAndPositionPopup();
  }

  function updateCharCount() {
    charCount.textContent = editor.value.length;
  }

  function escapeHtml(s) {
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // ---------- Ghost Text Mirror & Caret Tracking ----------
  function renderMirrorAndPositionPopup() {
    const pos = editor.selectionStart ?? editor.value.length;
    const val = editor.value;
    const before = val.slice(0, pos);
    const after = val.slice(pos);

    const showGhost =
      state.enableAuto &&
      state.chineseMode &&
      state.ghostText &&
      !state.rawPinyin &&
      pos === val.length; // Show inline ghost text cleanly at end of text or before newline

    let ghostHtml = "";
    if (state.enableAuto && state.chineseMode && state.ghostText && !state.rawPinyin) {
      // Only show inline ghost text when caret is at end of line/document so text never visually overlaps
      if (after === "" || after.startsWith("\n")) {
        ghostHtml = `<span class="mirror-ghost">${escapeHtml(state.ghostText)}</span><span class="mirror-tab-badge">Tab</span>`;
      }
    }

    mirror.innerHTML =
      escapeHtml(before) +
      `<span class="mirror-caret-anchor" id="caretAnchor"></span>` +
      ghostHtml +
      escapeHtml(after) +
      "\n";

    mirror.scrollTop = editor.scrollTop;

    const anchor = $("caretAnchor");
    if (anchor && !imePopup.classList.contains("hidden")) {
      const stageRect = editorStage.getBoundingClientRect();
      const anchorRect = anchor.getBoundingClientRect();
      let left = anchorRect.left - stageRect.left;
      let top = anchorRect.bottom - stageRect.top + 8;

      const popupW = Math.max(imePopup.offsetWidth, 350);
      const maxLeft = Math.max(8, stageRect.width - popupW - 12);
      left = Math.max(12, Math.min(left, maxLeft));

      const popupH = imePopup.offsetHeight || 86;
      if (top + popupH > stageRect.height - 52) {
        top = Math.max(8, anchorRect.top - stageRect.top - popupH - 8);
      }

      imePopup.style.left = `${left}px`;
      imePopup.style.top = `${top}px`;
    }
  }

  function setGhostStatus(msg) {
    ghostStatusText.textContent = msg;
  }

  // ---------- Core Query & Rendering ----------
  async function runQuery() {
    const seq = ++state.querySeq;
    const raw = state.rawPinyin;

    if (!raw) {
      if (state.pendingWords.length > 0) {
        await finalizePendingComposition();
        return;
      }
      state.candidates = [];
      state.displayPinyin = "";
      renderPopup();
      return;
    }

    state.assocMode = false;
    const ctx = getContextBeforeCaret();
    const data = await apiPost("/api/query", {
      raw,
      context: ctx,
      limit: 40,
      enable_memory: state.enableMem,
      enable_autocomplete: state.enableAuto,
    });

    if (seq !== state.querySeq) return; // Stale response

    state.displayPinyin = data.display_pinyin || raw;
    state.candidates = data.candidates || [];
    state.page = 0;
    state.activeIndex = 0;
    state.ghostText = data.ghost_text || "";
    state.ghostSource = data.ghost_source || "";

    $("statLatency").textContent = data.latency_ms ?? "--";
    imeLatency.textContent = `${data.latency_ms ?? 0}ms`;

    renderPopup();
    renderInspector(data);
    renderMirrorAndPositionPopup();
  }

  async function runContextPredict() {
    if (!state.enableAuto || !state.chineseMode || state.rawPinyin) {
      state.ghostText = "";
      state.ghostSource = "";
      state.associations = [];
      renderMirrorAndPositionPopup();
      return;
    }
    const ctx = getContextBeforeCaret();
    if (!ctx) {
      state.ghostText = "";
      state.ghostSource = "";
      state.associations = [];
      state.assocMode = false;
      renderPopup();
      renderMirrorAndPositionPopup();
      return;
    }
    const data = await apiPost("/api/predict", {
      context: ctx,
      limit: 8,
      enable_memory: state.enableMem,
    });
    state.ghostText = data.ghost_text || "";
    state.ghostSource = data.ghost_source || "";
    state.associations = data.associations || [];
    if (state.ghostText) {
      setGhostStatus(`智能补全就绪：「${state.ghostText}」(${state.ghostSource}) — 按 Tab 一键上屏`);
    } else {
      setGhostStatus("就绪 · 继续输入拼音");
    }
    renderMirrorAndPositionPopup();
  }

  function renderPopup() {
    const isTyping = state.rawPinyin.length > 0 || state.pendingWords.length > 0;
    const isAssoc = !isTyping && state.assocMode && state.associations.length > 0 && state.enableAuto;

    if (!isTyping && !isAssoc) {
      imePopup.classList.add("hidden");
      return;
    }

    imePopup.classList.remove("hidden");

    if (isAssoc) {
      imeAssocBadge.classList.remove("hidden");
      imePendingCn.textContent = "";
      imePinyinText.textContent = state.ghostSource || "上下文智能联想";
      imeGhostHint.classList.add("hidden");
      imeLatency.textContent = "Tab 采纳第1项";

      const items = state.associations.slice(0, PAGE_SIZE);
      pagerInfo.textContent = "1/1";
      btnPrevPage.disabled = true;
      btnNextPage.disabled = true;

      imeCandidates.innerHTML = items
        .map(
          (item, idx) => `
          <div class="ime-cand ${idx === state.activeIndex ? "active" : ""}" data-assoc-idx="${idx}">
            <span class="cand-num">${idx + 1}.</span>
            <span>${escapeHtml(item.word)}</span>
            <span class="cand-tag tag-${item.tag}">${escapeHtml(item.tag)}</span>
          </div>`
        )
        .join("");

      imeCandidates.querySelectorAll(".ime-cand").forEach((el) => {
        el.addEventListener("mousedown", (e) => {
          e.preventDefault();
          pickAssociation(Number(el.dataset.assocIdx));
        });
      });

      renderMirrorAndPositionPopup();
      return;
    }

    // Normal Pinyin candidate mode
    imeAssocBadge.classList.add("hidden");
    imePendingCn.textContent = state.pendingWords.map((p) => p.word).join("");
    imePinyinText.textContent = state.displayPinyin || state.rawPinyin;

    if (state.ghostText && state.enableAuto) {
      imeGhostHint.textContent = `补全预测: ${state.ghostText}`;
      imeGhostHint.classList.remove("hidden");
    } else {
      imeGhostHint.classList.add("hidden");
    }

    const total = state.candidates.length;
    const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
    state.page = Math.max(0, Math.min(state.page, totalPages - 1));
    const start = state.page * PAGE_SIZE;
    const pageItems = state.candidates.slice(start, start + PAGE_SIZE);

    if (state.activeIndex >= pageItems.length) {
      state.activeIndex = Math.max(0, pageItems.length - 1);
    }

    pagerInfo.textContent = `${state.page + 1}/${totalPages}`;
    btnPrevPage.disabled = state.page <= 0;
    btnNextPage.disabled = state.page >= totalPages - 1;

    if (pageItems.length === 0) {
      imeCandidates.innerHTML = `<span class="empty-hint" style="padding:4px 8px">无匹配词条，按 Enter 上屏原字母</span>`;
    } else {
      imeCandidates.innerHTML = pageItems
        .map((c, idx) => {
          const showTag = c.tag && c.tag !== "词库";
          const tagHtml = showTag
            ? `<span class="cand-tag tag-${c.tag}">${escapeHtml(c.tag)}</span>`
            : "";
          const hintHtml = c.completion_hint
            ? `<span class="cand-hint">${escapeHtml(c.completion_hint)}</span>`
            : "";
          return `
            <div class="ime-cand ${idx === state.activeIndex ? "active" : ""}" data-cand-idx="${start + idx}">
              <span class="cand-num">${idx + 1}.</span>
              <span>${escapeHtml(c.word)}</span>
              ${hintHtml}
              ${tagHtml}
            </div>`;
        })
        .join("");

      imeCandidates.querySelectorAll(".ime-cand").forEach((el) => {
        el.addEventListener("mousedown", (e) => {
          e.preventDefault();
          pickCandidate(Number(el.dataset.candIdx));
        });
      });
    }

    renderMirrorAndPositionPopup();
  }

  // ---------- Candidate Selection & Learning ----------
  async function pickCandidate(globalIdx) {
    const cand = state.candidates[globalIdx];
    if (!cand) return;

    const rawNoQuote = state.rawPinyin.replace(/'/g, "");
    const consumed = Math.min(cand.consumed_chars || rawNoQuote.length, rawNoQuote.length);
    const remaining = rawNoQuote.slice(consumed);

    state.pendingWords.push({ word: cand.word, pinyin: cand.pinyin });

    if (remaining.length > 0) {
      // Progressive selection (just like Sogou! e.g. pick '蚂蚁', keep 'yinli' to pick '引力')
      state.rawPinyin = remaining;
      await runQuery();
      return;
    }

    await finalizePendingComposition();
  }

  async function finalizePendingComposition() {
    if (state.pendingWords.length === 0) return;

    const prevCtx = editor.value.slice(0, editor.selectionStart ?? editor.value.length).slice(-32);
    const fullWord = state.pendingWords.map((p) => p.word).join("");
    const fullPinyin = state.pendingWords.map((p) => p.pinyin).join(" ");
    const isMultiStep = state.pendingWords.length >= 2;
    const lastPart = state.pendingWords[state.pendingWords.length - 1];

    // Insert full composed text into editor
    insertTextAtCaret(fullWord);

    // Reset composition buffer
    state.rawPinyin = "";
    state.displayPinyin = "";
    state.candidates = [];
    state.pendingWords = [];
    state.page = 0;
    state.activeIndex = 0;

    // Commit to backend (updates frequency, auto-learns composed new phrase, learns bigram, fetches associations)
    const res = await apiPost("/api/commit", {
      word: isMultiStep ? lastPart.word : fullWord,
      pinyin: isMultiStep ? lastPart.pinyin : fullPinyin,
      prev_context: isMultiStep ? prevCtx + fullWord.slice(0, fullWord.length - lastPart.word.length) : prevCtx,
      composed_phrase: isMultiStep ? [fullWord, fullPinyin] : null,
      enable_memory: state.enableMem,
      enable_autocomplete: state.enableAuto,
    });

    if (res.learned_events && res.learned_events.length > 0) {
      res.learned_events.forEach((ev) => pushLearningEvent(ev));
    }

    if (res.memory_stats) {
      updateMemoryStatsHeader(res.memory_stats);
    }

    state.ghostText = res.ghost_text || "";
    state.ghostSource = res.ghost_source || "";
    state.associations = res.associations || [];
    state.assocMode = state.enableAuto && state.associations.length > 0;
    state.activeIndex = 0;

    if (state.ghostText) {
      setGhostStatus(`智能补全：「${state.ghostText}」(${state.ghostSource}) — 按 Tab 或数字 1-5 接续上屏`);
    } else {
      setGhostStatus(`已上屏「${fullWord}」`);
    }

    renderPopup();
    renderMirrorAndPositionPopup();
    refreshMemoryTabIfActive();
  }

  async function pickAssociation(idx) {
    const item = state.associations[idx];
    if (!item) return;

    const prevCtx = editor.value.slice(0, editor.selectionStart ?? editor.value.length).slice(-32);
    insertTextAtCaret(item.word);

    const res = await apiPost("/api/commit", {
      word: item.word,
      pinyin: "",
      prev_context: prevCtx,
      enable_memory: state.enableMem,
      enable_autocomplete: state.enableAuto,
    });

    if (res.learned_events) {
      res.learned_events.forEach((ev) => pushLearningEvent(ev));
    }
    if (res.memory_stats) {
      updateMemoryStatsHeader(res.memory_stats);
    }

    state.ghostText = res.ghost_text || "";
    state.ghostSource = res.ghost_source || "";
    state.associations = res.associations || [];
    state.assocMode = state.enableAuto && state.associations.length > 0;
    state.activeIndex = 0;

    if (state.ghostText) {
      setGhostStatus(`继续补全：「${state.ghostText}」(${state.ghostSource}) — 按 Tab 采纳`);
    } else {
      setGhostStatus(`已补全「${item.word}」`);
    }

    renderPopup();
    renderMirrorAndPositionPopup();
    refreshMemoryTabIfActive();
  }

  function cancelComposition() {
    state.rawPinyin = "";
    state.displayPinyin = "";
    state.pendingWords = [];
    state.candidates = [];
    state.assocMode = false;
    state.page = 0;
    state.activeIndex = 0;
    renderPopup();
    renderMirrorAndPositionPopup();
  }

  function commitRawEnglish() {
    const text = state.pendingWords.map((p) => p.word).join("") + state.rawPinyin;
    if (text) insertTextAtCaret(text);
    cancelComposition();
  }

  // ---------- Keyboard Interception (Sogou Authentic Behavior) ----------
  editor.addEventListener("keydown", async (e) => {
    // Track pure Shift tap for CN/EN switch
    if (e.key === "Shift" && !e.ctrlKey && !e.altKey && !e.metaKey) {
      state.shiftDownAlone = true;
      return;
    } else {
      state.shiftDownAlone = false;
    }

    if (e.ctrlKey || e.altKey || e.metaKey) return;

    const isComposing = state.rawPinyin.length > 0 || state.pendingWords.length > 0;
    const isAssoc = !isComposing && state.assocMode && state.associations.length > 0 && state.enableAuto;

    // Tab: Accept inline Ghost Text completion!
    if (e.key === "Tab") {
      if (state.chineseMode && state.enableAuto && state.ghostText && !isComposing) {
        e.preventDefault();
        if (state.associations.length > 0) {
          await pickAssociation(0);
        } else {
          insertTextAtCaret(state.ghostText);
          state.ghostText = "";
          await runContextPredict();
        }
        return;
      }
      if (isComposing && state.candidates.length > 0) {
        // Tab while composing picks top completion or top candidate
        e.preventDefault();
        const compIdx = state.candidates.findIndex((c) => c.tag === "补全");
        await pickCandidate(compIdx >= 0 ? compIdx : state.page * PAGE_SIZE + state.activeIndex);
        return;
      }
      return;
    }

    // English mode passthrough
    if (!state.chineseMode) return;

    // Letters a-z -> enter or append to Pinyin buffer
    if (/^[a-zA-Z]$/.test(e.key)) {
      e.preventDefault();
      state.assocMode = false;
      state.rawPinyin += e.key.toLowerCase();
      await runQuery();
      return;
    }

    // Apostrophe `'` as syllable separator (e.g. xi'an) when composing
    if (e.key === "'" && isComposing) {
      e.preventDefault();
      if (!state.rawPinyin.endsWith("'")) {
        state.rawPinyin += "'";
        await runQuery();
      }
      return;
    }

    // Active composition controls
    if (isComposing) {
      if (e.key === "Backspace") {
        e.preventDefault();
        if (state.rawPinyin.length > 0) {
          state.rawPinyin = state.rawPinyin.slice(0, -1);
          if (state.rawPinyin.length === 0 && state.pendingWords.length > 0) {
            // Undo last progressive word pick
            const popped = state.pendingWords.pop();
            state.rawPinyin = popped.pinyin.replace(/\s+/g, "");
          }
          await runQuery();
        } else if (state.pendingWords.length > 0) {
          state.pendingWords.pop();
          renderPopup();
        }
        return;
      }

      if (e.key === "Escape") {
        e.preventDefault();
        cancelComposition();
        return;
      }

      if (e.key === "Enter") {
        e.preventDefault();
        commitRawEnglish();
        return;
      }

      if (e.key === " ") {
        e.preventDefault();
        if (state.candidates.length > 0) {
          await pickCandidate(state.page * PAGE_SIZE + state.activeIndex);
        } else {
          commitRawEnglish();
        }
        return;
      }

      if (/^[1-5]$/.test(e.key)) {
        e.preventDefault();
        const target = state.page * PAGE_SIZE + (Number(e.key) - 1);
        if (target < state.candidates.length) {
          await pickCandidate(target);
        }
        return;
      }

      if (e.key === "=" || e.key === "+" || e.key === "." || e.key === "PageDown") {
        e.preventDefault();
        changePage(1);
        return;
      }

      if (e.key === "-" || e.key === "_" || e.key === "," || e.key === "PageUp") {
        e.preventDefault();
        changePage(-1);
        return;
      }

      if (e.key === "ArrowRight" || e.key === "ArrowDown") {
        e.preventDefault();
        moveHighlight(1);
        return;
      }

      if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
        e.preventDefault();
        moveHighlight(-1);
        return;
      }

      // Pressing Chinese punctuation while composing commits candidate #1 + punctuation (classic Sogou!)
      if (state.cnPunct && (CN_PUNCT[e.key] || e.key === '"' || e.key === "'")) {
        e.preventDefault();
        if (state.candidates.length > 0) {
          await pickCandidate(state.page * PAGE_SIZE + state.activeIndex);
        } else {
          commitRawEnglish();
        }
        insertChinesePunct(e.key);
        return;
      }
      return;
    }

    // Association mode shortcuts (after committing a word)
    if (isAssoc) {
      if (/^[1-5]$/.test(e.key)) {
        const idx = Number(e.key) - 1;
        if (idx < state.associations.length) {
          e.preventDefault();
          await pickAssociation(idx);
          return;
        }
      }
      if (e.key === "Escape") {
        e.preventDefault();
        state.assocMode = false;
        renderPopup();
        return;
      }
      if (e.key === "ArrowRight" || e.key === "ArrowDown") {
        e.preventDefault();
        state.activeIndex = (state.activeIndex + 1) % Math.min(PAGE_SIZE, state.associations.length);
        renderPopup();
        return;
      }
      if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
        e.preventDefault();
        const len = Math.min(PAGE_SIZE, state.associations.length);
        state.activeIndex = (state.activeIndex - 1 + len) % len;
        renderPopup();
        return;
      }
      // Any other editing key closes association popup
      state.assocMode = false;
      renderPopup();
    }

    // Smart Chinese punctuation when not composing
    if (state.cnPunct && (CN_PUNCT[e.key] || e.key === '"' || e.key === "'")) {
      e.preventDefault();
      insertChinesePunct(e.key);
      state.ghostText = "";
      state.associations = [];
      renderMirrorAndPositionPopup();
    }
  });

  editor.addEventListener("keyup", (e) => {
    if (e.key === "Shift" && state.shiftDownAlone) {
      state.shiftDownAlone = false;
      toggleLangMode();
    }
  });

  editor.addEventListener("input", () => {
    updateCharCount();
    renderMirrorAndPositionPopup();
    if (!state.rawPinyin) {
      runContextPredict();
    }
  });

  editor.addEventListener("scroll", () => {
    mirror.scrollTop = editor.scrollTop;
    renderMirrorAndPositionPopup();
  });

  editor.addEventListener("click", () => {
    if (!state.rawPinyin) {
      state.assocMode = false;
      renderPopup();
      runContextPredict();
    }
  });

  function insertChinesePunct(key) {
    let ch = CN_PUNCT[key];
    if (key === '"') {
      ch = state.quoteDoubleOpen ? "“" : "”";
      state.quoteDoubleOpen = !state.quoteDoubleOpen;
    } else if (key === "'") {
      ch = state.quoteSingleOpen ? "‘" : "’";
      state.quoteSingleOpen = !state.quoteSingleOpen;
    }
    if (ch) insertTextAtCaret(ch);
  }

  function changePage(delta) {
    const totalPages = Math.max(1, Math.ceil(state.candidates.length / PAGE_SIZE));
    const next = state.page + delta;
    if (next >= 0 && next < totalPages) {
      state.page = next;
      state.activeIndex = 0;
      renderPopup();
    }
  }

  function moveHighlight(delta) {
    const start = state.page * PAGE_SIZE;
    const pageLen = Math.min(PAGE_SIZE, state.candidates.length - start);
    if (pageLen <= 0) return;
    let next = state.activeIndex + delta;
    if (next >= pageLen) {
      changePage(1);
    } else if (next < 0) {
      if (state.page > 0) {
        state.page -= 1;
        state.activeIndex = PAGE_SIZE - 1;
        renderPopup();
      }
    } else {
      state.activeIndex = next;
      renderPopup();
    }
  }

  btnPrevPage.addEventListener("mousedown", (e) => {
    e.preventDefault();
    changePage(-1);
  });
  btnNextPage.addEventListener("mousedown", (e) => {
    e.preventDefault();
    changePage(1);
  });

  // ---------- Status Bar Toggles & Themes ----------
  function toggleLangMode() {
    if (state.rawPinyin) commitRawEnglish();
    state.chineseMode = !state.chineseMode;
    btnToggleLang.textContent = state.chineseMode ? "中" : "英";
    btnToggleLang.classList.toggle("active", state.chineseMode);
    if (!state.chineseMode) {
      state.assocMode = false;
      state.ghostText = "";
      renderPopup();
      renderMirrorAndPositionPopup();
      setGhostStatus("已切换为英文输入模式（轻按 Shift 切回中文）");
    } else {
      setGhostStatus("已切换为中文拼音模式");
    }
    editor.focus();
  }

  btnToggleLang.addEventListener("click", toggleLangMode);

  btnTogglePunct.addEventListener("click", () => {
    state.cnPunct = !state.cnPunct;
    btnTogglePunct.textContent = state.cnPunct ? "。，" : ".,";
    btnTogglePunct.classList.toggle("active", state.cnPunct);
    editor.focus();
  });

  btnToggleAuto.addEventListener("click", () => {
    state.enableAuto = !state.enableAuto;
    btnToggleAuto.classList.toggle("active", state.enableAuto);
    if (!state.enableAuto) {
      state.ghostText = "";
      state.assocMode = false;
      renderPopup();
      renderMirrorAndPositionPopup();
    } else {
      runContextPredict();
    }
    editor.focus();
  });

  btnToggleMem.addEventListener("click", () => {
    state.enableMem = !state.enableMem;
    btnToggleMem.classList.toggle("active", state.enableMem);
    if (state.rawPinyin) runQuery();
    editor.focus();
  });

  document.querySelectorAll("[data-theme-pick]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const theme = btn.dataset.themePick;
      document.documentElement.setAttribute("data-theme", theme);
      document.querySelectorAll("[data-theme-pick]").forEach((b) => b.classList.toggle("active", b === btn));
      localStorage.setItem("sogou_ime_theme", theme);
    });
  });

  const savedTheme = localStorage.getItem("sogou_ime_theme");
  if (savedTheme) {
    document.documentElement.setAttribute("data-theme", savedTheme);
    document.querySelectorAll("[data-theme-pick]").forEach((b) =>
      b.classList.toggle("active", b.dataset.themePick === savedTheme)
    );
  }

  // ---------- Inspector & Memory Panel ----------
  function renderInspector(data) {
    const segs = data.segmentations || [];
    const segEl = $("segChips");
    if (segs.length === 0) {
      segEl.innerHTML = `<span class="empty-hint">输入拼音后实时显示切分方案...</span>`;
    } else {
      segEl.innerHTML = segs
        .map((s, i) => `<span class="seg-chip ${i === 0 ? "primary" : ""}">${escapeHtml(s)}</span>`)
        .join("");
    }

    const cands = (data.candidates || []).slice(0, 6);
    const scoreEl = $("scoreList");
    if (cands.length === 0) {
      scoreEl.innerHTML = `<span class="empty-hint">暂无候选词</span>`;
      return;
    }

    const maxScore = Math.max(1, ...cands.map((c) => c.score));
    scoreEl.innerHTML = cands
      .map((c) => {
        const b = Math.max(0, c.breakdown?.base || 0);
        const m = Math.max(0, c.breakdown?.memory || 0);
        const x = Math.max(0, c.breakdown?.context || 0);
        const sum = Math.max(0.01, b + m + x);
        const pctBase = Math.round((b / sum) * (c.score / maxScore) * 100);
        const pctMem = Math.round((m / sum) * (c.score / maxScore) * 100);
        const pctCtx = Math.round((x / sum) * (c.score / maxScore) * 100);
        return `
          <div class="score-row">
            <div class="score-row-top">
              <span class="score-word">
                ${escapeHtml(c.word)}
                <span class="cand-tag tag-${c.tag}">${escapeHtml(c.tag)}</span>
              </span>
              <span class="score-total">${c.score}</span>
            </div>
            <div class="score-bars">
              <div class="bar-base" style="width:${Math.max(4, pctBase)}%" title="词库分"></div>
              <div class="bar-mem" style="width:${pctMem}%" title="记忆分"></div>
              <div class="bar-ctx" style="width:${pctCtx}%" title="上下文分"></div>
            </div>
            <div class="score-meta">
              <span>词库: ${b.toFixed(1)}</span>
              <span>记忆: +${m.toFixed(1)}</span>
              <span>上下文: +${x.toFixed(1)}</span>
              ${c.user_freq ? `<span>用过 ${c.user_freq} 次</span>` : ""}
            </div>
          </div>`;
      })
      .join("");
  }

  function pushLearningEvent(text) {
    const timeStr = new Date().toLocaleTimeString("zh-CN", { hour12: false });
    state.events.unshift(`[${timeStr}] ${text}`);
    if (state.events.length > 20) state.events.pop();
    $("eventFeed").innerHTML = state.events
      .map((ev) => `<li class="event-item">${escapeHtml(ev)}</li>`)
      .join("");
  }

  function updateMemoryStatsHeader(memStats) {
    $("statMemCount").textContent = memStats.total_words ?? 0;
    $("statBigramCount").textContent = memStats.total_bigrams ?? 0;
  }

  async function loadMemoryPanel() {
    const data = await apiGet("/api/memory");
    updateMemoryStatsHeader(data.stats);
    $("memWordCount").textContent = data.words.length;
    $("memBigramCount").textContent = data.bigrams.length;

    const wEl = $("memWordsList");
    if (data.words.length === 0) {
      wEl.innerHTML = `<span class="empty-hint">暂无用户记忆词条，打字选词后会自动积累</span>`;
    } else {
      wEl.innerHTML = data.words
        .map(
          (w) => `
          <div class="mem-item">
            <div class="mem-item-left">
              <div class="mem-word-line">
                <span>${escapeHtml(w.word)}</span>
                ${w.is_custom ? `<span class="cand-tag tag-新词">新词</span>` : `<span class="cand-tag tag-记忆">记忆</span>`}
              </div>
              <span class="mem-py">${escapeHtml(w.pinyin)} (${escapeHtml(w.initials)})</span>
            </div>
            <div class="mem-right">
              <span class="mem-freq">×${w.freq}</span>
              <button class="btn-del" data-del-word="${escapeHtml(w.word)}" data-del-py="${escapeHtml(w.pinyin)}" title="删除">✕</button>
            </div>
          </div>`
        )
        .join("");

      wEl.querySelectorAll(".btn-del").forEach((btn) => {
        btn.addEventListener("click", async () => {
          await apiDelete("/api/memory/word", {
            word: btn.dataset.delWord,
            pinyin: btn.dataset.delPy,
          });
          await loadMemoryPanel();
        });
      });
    }

    const bEl = $("memBigramsList");
    if (data.bigrams.length === 0) {
      bEl.innerHTML = `<span class="empty-hint">连续输入词语后，会自动记住「前词 → 后词」搭配</span>`;
    } else {
      bEl.innerHTML = data.bigrams
        .map(
          (b) => `
          <div class="mem-item">
            <div class="mem-item-left">
              <div class="mem-word-line">${escapeHtml(b.prev_word)} → ${escapeHtml(b.next_word)}</div>
            </div>
            <div class="mem-right">
              <span class="mem-freq">×${b.freq}</span>
              <button class="btn-del" data-del-prev="${escapeHtml(b.prev_word)}" data-del-next="${escapeHtml(b.next_word)}" title="删除">✕</button>
            </div>
          </div>`
        )
        .join("");

      bEl.querySelectorAll(".btn-del").forEach((btn) => {
        btn.addEventListener("click", async () => {
          await apiDelete("/api/memory/bigram", {
            prev_word: btn.dataset.delPrev,
            next_word: btn.dataset.delNext,
          });
          await loadMemoryPanel();
        });
      });
    }
  }

  function refreshMemoryTabIfActive() {
    if ($("tabMemory").classList.contains("active")) {
      loadMemoryPanel();
    }
  }

  // Panel tabs switching
  document.querySelectorAll(".panel-tab").forEach((tabBtn) => {
    tabBtn.addEventListener("click", () => {
      const target = tabBtn.dataset.tab;
      document.querySelectorAll(".panel-tab").forEach((b) => b.classList.toggle("active", b === tabBtn));
      $("tabInspector").classList.toggle("active", target === "inspector");
      $("tabMemory").classList.toggle("active", target === "memory");
      if (target === "memory") loadMemoryPanel();
    });
  });

  // Add custom word form
  $("addWordForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const word = $("customWord").value.trim();
    const pinyin = $("customPinyin").value.trim();
    if (!word || !pinyin) return;
    await apiPost("/api/memory/word", { word, pinyin });
    $("customWord").value = "";
    $("customPinyin").value = "";
    pushLearningEvent(`手动添加新词: ${word} (${pinyin})`);
    await loadMemoryPanel();
  });

  // Clear all memory
  $("btnClearMemory").addEventListener("click", async () => {
    if (!confirm("确定清空所有用户记忆词和上下文联想吗？")) return;
    await apiPost("/api/memory/clear", {});
    pushLearningEvent("已清空所有用户记忆");
    await loadMemoryPanel();
  });

  // ---------- Quick Demo Buttons ----------
  document.querySelectorAll("[data-demo]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const kind = btn.dataset.demo;
      if (!state.chineseMode) toggleLangMode();
      cancelComposition();
      editor.focus();

      if (kind === "sentence") {
        state.rawPinyin = "woxiangqubeijinglvyou";
        await runQuery();
        setGhostStatus("整句演示：按 空格 直接上屏「我想去北京旅游」");
      } else if (kind === "jianpin") {
        state.rawPinyin = "jttq";
        await runQuery();
        setGhostStatus("简拼补全：按 1 选「今天天气」，按 2 选自动补全「今天天气真好」");
      } else if (kind === "prefix") {
        state.rawPinyin = "zhongg";
        await runQuery();
        setGhostStatus("前缀补全：未打完的拼音自动匹配「中国」及联想词");
      } else if (kind === "context") {
        editor.value = "今天天";
        editor.selectionStart = editor.selectionEnd = editor.value.length;
        updateCharCount();
        await runContextPredict();
        state.assocMode = state.associations.length > 0;
        renderPopup();
        renderMirrorAndPositionPopup();
      } else if (kind === "learn") {
        state.rawPinyin = "mayiyinli";
        await runQuery();
        setGhostStatus("新造词演练：连续选「蚂蚁」再选「引力」，下次输 myyl 或 mayiyinli 直接首位命中「蚂蚁引力」！");
      }
    });
  });

  $("btnClearEditor").addEventListener("click", () => {
    editor.value = "";
    cancelComposition();
    state.ghostText = "";
    state.associations = [];
    updateCharCount();
    renderMirrorAndPositionPopup();
    setGhostStatus("画布已清空");
    editor.focus();
  });

  // ---------- Init ----------
  async function init() {
    try {
      const stats = await apiGet("/api/stats");
      $("statDictCount").textContent = (stats.dictionary?.total_entries || 0).toLocaleString();
      updateMemoryStatsHeader(stats.memory || {});
    } catch (err) {
      setGhostStatus("连接引擎失败，请确认 server.py 正在运行");
    }
    editor.focus();
    renderMirrorAndPositionPopup();
  }

  init();
})();
