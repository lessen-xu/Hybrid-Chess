/* The server owns the game. Draft settings and board interaction stay local. */
(() => {
  const I = window.HybridI18n,
    BR = window.BoardRenderer,
    $ = (id) => document.getElementById(id);
  const svg = $("boardSvg"),
    t = I.t;
  let catalog = null,
    agents = [],
    current = null,
    draft = null,
    editing = true,
    setupCollapsed = false;
  let preview = null,
    previewPending = false,
    previewSequence = 0,
    previewTimer = null;
  let connected = false,
    busy = false,
    activity = "",
    errorCode = "",
    flipped = false,
    selected = null;
  let cursor = { x: 4, y: 0 },
    keyboardFocus = false,
    promotionMoves = [],
    confirmKind = "",
    confirmResolve = null;
  function element(tag, className = "", value = "") {
    const node = document.createElement(tag);
    node.className = className;
    node.textContent = value;
    return node;
  }
  function icon(ch) {
    return BR.CHESS_PIECES[ch]?.icon || BR.XIANGQI_PIECES[ch]?.ch || ch;
  }
  const activeVariant = () =>
    editing ? preview?.variant || draft?.variant || {} : current?.variant || {};
  const boardAscii = () =>
    editing ? preview?.board_ascii || "" : current?.board_ascii || "";
  const opts = () => ({
    ...BR.defaultOpts(),
    flipped,
    variant: activeVariant(),
  });
  const myTurn = () =>
    current &&
    !current.game_over &&
    current.side_to_move === current.human_side;
  const aiTurn = () => current && !current.game_over && !myTurn();
  const guard = () =>
    current
      ? { session_id: current.session_id, revision: current.revision }
      : {};
  const isSame = (a, b) =>
    Object.keys(catalog.defaults).every((key) => a?.[key] === b?.[key]);
  const presetFor = (v) => catalog?.presets.find((p) => isSame(v, p.variant));
  async function request(path, body) {
    let response;
    try {
      response = await fetch(
        "/api/" + path,
        body === undefined
          ? { cache: "no-store" }
          : {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(body),
            },
      );
    } catch {
      throw new Error("network");
    }
    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error("server_error");
    }
    if (!response.ok) throw new Error(data.error || "server_error");
    return data;
  }
  async function readState() {
    try {
      return await request("state");
    } catch (error) {
      if (error.message === "no_game") return null;
      throw error;
    }
  }
  function buildControls() {
    $("presets").replaceChildren();
    catalog.presets.forEach((p, index) => {
      const button = element("button", "preset");
      button.dataset.preset = p.id;
      button.append(
        element("span", "preset-number", String(index + 1).padStart(2, "0")),
        element("span", "preset-name", I.localized(p.name)),
      );
      button.title = I.localized(p.description);
      button.addEventListener("click", () => {
        if (busy) return;
        draft.variant = { ...p.variant };
        schedulePreview();
      });
      $("presets").append(button);
    });
    $("customRules").replaceChildren();
    for (const group of ["chess", "xiangqi", "movement"]) {
      const fieldset = element("fieldset", "rule-group");
      fieldset.append(element("legend", "", t("group_" + group)));
      for (const rule of catalog.rules.filter((r) => r.group === group)) {
        const label = element("label", "rule-option"),
          copy = element("span");
        copy.append(
          element("strong", "", I.localized(rule.name)),
          element("small", "", I.localized(rule.description)),
        );
        const input = document.createElement("input");
        input.type = "checkbox";
        input.dataset.rule = rule.key;
        input.addEventListener("change", () => {
          draft.variant[rule.key] = input.checked;
          if (draft.variant.no_promotion)
            draft.variant.no_queen_promotion = false;
          schedulePreview();
        });
        label.append(copy, input);
        fieldset.append(label);
      }
      $("customRules").append(fieldset);
    }
    $("aiSelect").replaceChildren(
      ...agents.map((a) => {
        const option = element("option", "", I.localized(a.name));
        option.value = a.id;
        return option;
      }),
    );
  }
  function renderSetup() {
    document.querySelectorAll("[data-side]").forEach((button) => {
      button.disabled = busy || !draft || !connected;
    });
    $("aiSelect").disabled = busy || !draft || !connected;
    if (!catalog || !draft) return;
    const chosen = presetFor(draft.variant);
    document.querySelectorAll("[data-preset]").forEach((button) => {
      const active = button.dataset.preset === chosen?.id;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
      button.disabled = busy || !connected;
    });
    document.querySelectorAll("[data-rule]").forEach((input) => {
      input.checked = !!draft.variant[input.dataset.rule];
      input.disabled =
        busy ||
        !connected ||
        (input.dataset.rule === "no_queen_promotion" &&
          draft.variant.no_promotion);
    });
    $("presetDescription").textContent = chosen
      ? I.localized(chosen.description)
      : t("customDesc");
    document.querySelectorAll("[data-side]").forEach((button) => {
      const active = button.dataset.side === draft.human_side;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
      button.disabled = busy || !connected;
    });
    $("aiSelect").value = draft.ai_agent;
    $("aiSelect").disabled = busy || !connected;
    const opponent = agents.find((a) => a.id === draft.ai_agent);
    $("aiHint").textContent = opponent?.seconds
      ? t("thinkBudget", { seconds: opponent.seconds })
      : t("practiceHint");
    $("btnStart").disabled = busy || !connected || previewPending || !preview;
    $("btnCancelSetup").hidden = !current;
    $("btnCancelSetup").disabled = busy;
  }
  function draw() {
    BR.drawBoard(svg, opts());
    BR.drawCoordLabels(svg, opts());
    BR.clearHighlights(svg);
    BR.drawPieces(svg, boardAscii(), opts());
    const last = !editing && current?.moves.at(-1);
    if (last)
      BR.highlightMove(
        svg,
        { x: last.fx, y: last.fy },
        { x: last.tx, y: last.ty },
        opts(),
      );
    const grid = BR.parseAsciiBoard(boardAscii());
    if (!editing && current?.in_check) {
      const royal = current.side_to_move === "chess" ? "K" : "g";
      for (let y = 0; y < 10; y++)
        for (let x = 0; x < 9; x++)
          if (grid[y][x] === royal)
            BR.highlightSquares(svg, [{ x, y }], "check", opts());
    }
    if (selected && !editing) {
      BR.highlightSquares(svg, [selected], "selected", opts());
      const destinations = new Map();
      for (const move of current.legal_moves.filter(
        (m) => m.fx === selected.x && m.fy === selected.y,
      ))
        destinations.set(move.tx + "," + move.ty, {
          x: move.tx,
          y: move.ty,
          capture: !!grid[move.ty][move.tx],
        });
      BR.highlightSquares(
        svg,
        [...destinations.values()],
        "legal-move",
        opts(),
      );
    }
    if (keyboardFocus) BR.highlightSquares(svg, [cursor], "focus", opts());
  }
  function renderPlayers() {
    const human = editing
      ? draft?.human_side || "chess"
      : current?.human_side || "chess";
    const agentId = editing ? draft?.ai_agent : current?.ai_agent;
    const opponent = agents.find((a) => a.id === agentId);
    for (const [position, side] of [
      ["Top", flipped ? "chess" : "xiangqi"],
      ["Bottom", flipped ? "xiangqi" : "chess"],
    ]) {
      const you = side === human;
      $("avatar" + position).textContent = side === "chess" ? "♔" : "将";
      $("avatar" + position).classList.toggle("xiangqi", side === "xiangqi");
      $("player" + position + "Name").textContent = t(side);
      $("player" + position + "Detail").textContent = you
        ? t("you")
        : t("ai") + " · " + (opponent ? I.localized(opponent.name) : "");
      const status = $("player" + position + "Status");
      status.replaceChildren();
      status.classList.remove("status-check");
      let label = "";
      if (editing && you)
        label = previewPending ? t("previewLoading") : t("preview");
      else if (!editing && current?.game_over && you) label = t("gameOver");
      else if (!editing && current?.side_to_move === side) {
        label = busy
          ? t(activity || "thinking")
          : t(you ? "yourTurn" : "aiTurn");
        if (current.in_check) {
          label = t("check") + " · " + label;
          status.classList.add("status-check");
        }
        status.append(element("span", "turn-dot"));
      }
      status.append(document.createTextNode(label));
    }
    const statusText = editing
      ? t("ready")
      : current?.game_over
        ? t(current.result_code)
        : busy
          ? t(activity)
          : t(myTurn() ? "yourTurn" : "aiTurn");
    if ($("liveStatus").textContent !== statusText)
      $("liveStatus").textContent = statusText;
    $("plyLabel").textContent = t("turnCount", {
      count: editing ? 0 : current?.ply || 0,
    });
  }
  function renderHelp() {
    const ch = selected
      ? BR.parseAsciiBoard(boardAscii())[selected.y][selected.x]
      : null;
    if (ch) {
      $("pieceHelpTitle").textContent = t("selectedPiece", {
        piece: I.pieceName(ch),
        square: "abcdefghi"[selected.x] + (selected.y + 1),
      });
      $("pieceHelpText").textContent = I.pieceDescription(ch, activeVariant());
    } else {
      $("pieceHelpTitle").textContent = t(
        editing ? "beforeStart" : current?.game_over ? "gameOver" : "boardHint",
      );
      $("pieceHelpText").textContent = t("moveHint");
    }
  }
  function renderSession() {
    if (!current || !catalog) return;
    const p = presetFor(current.variant);
    $("sessionRuleName").textContent = p
      ? I.localized(p.name)
      : t("customName");
    $("sessionRuleDescription").textContent = p
      ? I.localized(p.description)
      : t("customDesc");
    const changed = catalog.rules.filter(
      (r) => current.variant[r.key] !== catalog.defaults[r.key],
    );
    $("ruleTags").replaceChildren(
      ...changed.map((r) =>
        element(
          "span",
          "rule-tag",
          r.key === "flying_general"
            ? t("disabledFlying")
            : I.localized(r.name),
        ),
      ),
    );
    if (!changed.length)
      $("ruleTags").append(element("span", "small muted", t("unchanged")));
    $("historyCount").textContent = String(current.moves.length);
    $("moveHistory").replaceChildren();
    if (!current.moves.length)
      $("moveHistory").append(
        element(
          "p",
          "history-empty",
          t(current.human_side === "chess" ? "noMoves" : "noMovesAI"),
        ),
      );
    for (let i = 0; i < current.moves.length; i += 2) {
      const row = element("div", "move-row");
      row.append(element("span", "number", String(i / 2 + 1) + "."));
      for (let j = 0; j < 2; j++) {
        const move = current.moves[i + j],
          node = element("span", j ? "xq-move" : "", move?.notation || "—");
        if (i + j === current.moves.length - 1) node.classList.add("last-move");
        row.append(node);
      }
      $("moveHistory").append(row);
    }
    $("resultPanel").hidden = !current.game_over;
    $("resultTitle").textContent = t(current.result_code);
    $("resultReason").textContent = current.reason_code
      ? t(current.reason_code)
      : "";
    $("btnAgain").disabled = busy;
    $("btnChangeRules").disabled = busy;
  }
  function renderDialogs() {
    if (confirmKind) {
      $("confirmTitle").textContent = t(
        confirmKind === "new" ? "confirmNew" : "confirmResign",
      );
      $("confirmBody").textContent = t(
        confirmKind === "new" ? "confirmNewBody" : "confirmResignBody",
      );
    }
    $("promotionOptions").style.gridTemplateColumns =
      "repeat(" + (promotionMoves.length || 4) + ", 1fr)";
    $("promotionOptions").replaceChildren(
      ...promotionMoves.map((move) => {
        const ch = { QUEEN: "Q", ROOK: "R", BISHOP: "B", KNIGHT: "N" }[
          move.promotion
        ];
        const button = element("button", "promotion-choice");
        button.append(
          element("span", "symbol", icon(ch)),
          element("span", "", I.pieceName(ch)),
        );
        button.addEventListener("click", () => {
          $("promotionDialog").close();
          promotionMoves = [];
          run("move", move);
        });
        return button;
      }),
    );
    $("guidePieces").replaceChildren();
    for (const ch of [
      "K",
      "Q",
      "R",
      "B",
      "N",
      "P",
      "g",
      "a",
      "e",
      "h",
      "c",
      "n",
      "s",
      "q",
    ]) {
      if (ch === "q" && !activeVariant().xq_queen) continue;
      const row = element("div", "guide-piece"),
        copy = element("div");
      copy.append(
        element(
          "b",
          "",
          t(BR.CHESS_PIECES[ch] ? "chess" : "xiangqi") +
            " · " +
            I.pieceName(ch),
        ),
        element("p", "", I.pieceDescription(ch, activeVariant())),
      );
      row.append(element("span", "guide-symbol", icon(ch)), copy);
      $("guidePieces").append(row);
    }
  }
  function render() {
    I.apply();
    document.title = "Hybrid Chess · " + t("play");
    $("sidebarTitle").textContent = t(editing ? "setup" : "ongoing");
    if (!editing && current?.game_over)
      $("sidebarTitle").textContent = t("gameOver");
    $("setupPanel").hidden = !editing || setupCollapsed;
    $("sessionPanel").hidden = editing;
    $("btnNewGame").hidden = editing;
    $("btnNewGame").disabled = busy;
    $("btnToggleSetup").hidden = !editing;
    $("btnToggleSetup").setAttribute("aria-expanded", String(!setupCollapsed));
    $("btnToggleSetup").textContent = t(setupCollapsed ? "setup" : "close");
    $("btnUndo").disabled = busy || editing || !current?.can_undo;
    $("btnResign").disabled = busy || editing || !current || current.game_over;
    $("errorNotice").hidden = !errorCode;
    $("errorText").textContent = errorCode ? t(errorCode) : "";
    $("btnRetry").disabled = busy;
    $("btnRetry").textContent = t(
      !connected ? "reconnect" : aiTurn() && !editing ? "retry" : "sync",
    );
    renderSetup();
    draw();
    renderPlayers();
    renderHelp();
    renderSession();
    renderDialogs();
  }
  function schedulePreview() {
    selected = null;
    previewPending = true;
    const sequence = ++previewSequence;
    clearTimeout(previewTimer);
    render();
    previewTimer = setTimeout(() => updatePreview(sequence), 150);
  }
  async function updatePreview(sequence = ++previewSequence) {
    try {
      const data = await request("preview", { variant: draft.variant });
      if (sequence !== previewSequence || !editing) return;
      preview = data;
      draft.variant = { ...data.variant };
      connected = true;
      errorCode = "";
    } catch (error) {
      if (sequence !== previewSequence) return;
      preview = null;
      errorCode = error.message;
      if (error.message === "network") connected = false;
    } finally {
      if (sequence === previewSequence) {
        previewPending = false;
        render();
      }
    }
  }
  function accept(data) {
    current = data;
    selected = null;
    editing = false;
    setupCollapsed = false;
    errorCode = "";
    connected = true;
    previewSequence++;
    previewPending = false;
  }
  async function run(action, body = {}) {
    if (busy) return;
    busy = true;
    activity = action === "ai_move" ? "thinking" : "sending";
    errorCode = "";
    render();
    try {
      accept(await request(action, { ...body, ...guard() }));
      render();
      $("moveHistory").scrollTop = $("moveHistory").scrollHeight;
      if (["new", "move"].includes(action) && aiTurn()) {
        activity = "thinking";
        render();
        accept(await request("ai_move", guard()));
      }
    } catch (error) {
      const original = error.message;
      try {
        current = await readState();
        connected = true;
        if (current) {
          editing = false;
          selected = null;
        } else {
          editing = true;
          schedulePreview();
        }
      } catch {
        connected = false;
      }
      errorCode = connected ? original : "network";
    } finally {
      busy = false;
      activity = "";
      render();
      $("moveHistory").scrollTop = $("moveHistory").scrollHeight;
    }
  }
  function confirmAction(kind) {
    confirmKind = kind;
    renderDialogs();
    $("confirmDialog").showModal();
    $("confirmCancel").focus();
    return new Promise((resolve) => {
      confirmResolve = resolve;
    });
  }
  async function start() {
    if (busy || previewPending || !preview) return;
    if (
      current &&
      current.ply > 0 &&
      !current.game_over &&
      !(await confirmAction("new"))
    )
      return;
    flipped = draft.human_side === "xiangqi";
    I.preference.set("settings", draft);
    await run("new", { ...draft, variant: { ...draft.variant } });
  }
  function openSetup() {
    if (busy) return;
    if (current)
      draft = {
        human_side: current.human_side,
        ai_agent: current.ai_agent,
        variant: { ...current.variant },
      };
    editing = true;
    setupCollapsed = false;
    selected = null;
    flipped = draft.human_side === "xiangqi";
    schedulePreview();
    $("sidebarTitle").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
  function chooseSquare(x, y) {
    if (busy || editing || !myTurn() || promotionMoves.length) return;
    cursor = { x, y };
    const grid = BR.parseAsciiBoard(current.board_ascii),
      ch = grid[y][x];
    if (selected) {
      const moves = current.legal_moves.filter(
        (m) =>
          m.fx === selected.x &&
          m.fy === selected.y &&
          m.tx === x &&
          m.ty === y,
      );
      if (moves.length) {
        if (moves.length > 1) {
          promotionMoves = moves;
          renderDialogs();
          $("promotionDialog").showModal();
          $("promotionOptions").querySelector("button")?.focus();
        } else run("move", moves[0]);
        return;
      }
    }
    const own =
      ch &&
      (current.human_side === "chess"
        ? !!BR.CHESS_PIECES[ch]
        : !!BR.XIANGQI_PIECES[ch]);
    selected =
      own && !(selected?.x === x && selected?.y === y) ? { x, y } : null;
    draw();
    renderHelp();
  }
  async function initialize() {
    if (busy) return;
    busy = true;
    activity = "loading";
    render();
    try {
      const [rules, available, state] = await Promise.all([
        request("variants"),
        request("agents"),
        readState(),
      ]);
      catalog = rules;
      agents = available.agents;
      current = state;
      connected = true;
      errorCode = "";
      const saved = I.preference.get("settings", {});
      draft = {
        human_side: ["chess", "xiangqi"].includes(saved.human_side)
          ? saved.human_side
          : "chess",
        ai_agent: agents.some((a) => a.id === saved.ai_agent)
          ? saved.ai_agent
          : "ab_d1",
        variant: { ...catalog.defaults, ...saved.variant },
      };
      editing = !current;
      if (current) {
        draft = {
          human_side: current.human_side,
          ai_agent: current.ai_agent,
          variant: { ...current.variant },
        };
      }
      flipped = draft.human_side === "xiangqi";
      buildControls();
      if (editing) {
        try {
          preview = await request("preview", { variant: draft.variant });
        } catch (error) {
          if (error.message !== "invalid_variant") throw error;
          draft.variant = { ...catalog.defaults };
          preview = await request("preview", { variant: draft.variant });
        }
        draft.variant = { ...preview.variant };
      }
    } catch (error) {
      errorCode = error.message;
      connected = false;
    } finally {
      busy = false;
      activity = "";
      render();
    }
    if (connected && aiTurn() && !editing) await run("ai_move");
  }
  document.querySelectorAll("[data-side]").forEach((button) =>
    button.addEventListener("click", () => {
      if (busy || !draft) return;
      draft.human_side = button.dataset.side;
      flipped = draft.human_side === "xiangqi";
      render();
    }),
  );
  $("aiSelect").addEventListener("change", () => {
    draft.ai_agent = $("aiSelect").value;
    renderSetup();
  });
  $("btnStart").addEventListener("click", start);
  $("btnNewGame").addEventListener("click", openSetup);
  $("btnChangeRules").addEventListener("click", openSetup);
  $("btnAgain").addEventListener("click", () => {
    draft = {
      human_side: current.human_side,
      ai_agent: current.ai_agent,
      variant: { ...current.variant },
    };
    flipped = draft.human_side === "xiangqi";
    run("new", draft);
  });
  $("btnCancelSetup").addEventListener("click", () => {
    editing = false;
    selected = null;
    previewSequence++;
    previewPending = false;
    flipped = current.human_side === "xiangqi";
    render();
  });
  $("btnToggleSetup").addEventListener("click", () => {
    setupCollapsed = !setupCollapsed;
    render();
  });
  $("btnFlip").addEventListener("click", () => {
    flipped = !flipped;
    draw();
    renderPlayers();
  });
  $("btnUndo").addEventListener("click", () => run("undo"));
  $("btnResign").addEventListener("click", async () => {
    if (!busy && (await confirmAction("resign"))) run("resign");
  });
  $("btnRetry").addEventListener("click", async () => {
    if (busy) return;
    if (connected && current && aiTurn() && !editing) await run("ai_move");
    else await initialize();
  });
  $("confirmCancel").addEventListener("click", () =>
    $("confirmDialog").close("cancel"),
  );
  $("confirmAccept").addEventListener("click", () =>
    $("confirmDialog").close("yes"),
  );
  $("confirmDialog").addEventListener(
    "cancel",
    () => ($("confirmDialog").returnValue = "cancel"),
  );
  $("confirmDialog").addEventListener("close", () => {
    const resolve = confirmResolve;
    confirmResolve = null;
    confirmKind = "";
    resolve?.($("confirmDialog").returnValue === "yes");
  });
  $("promotionCancel").addEventListener("click", () =>
    $("promotionDialog").close(),
  );
  $("promotionDialog").addEventListener("close", () => {
    promotionMoves = [];
    draw();
    renderHelp();
  });
  $("btnHelp").addEventListener("click", () => {
    renderDialogs();
    $("helpDialog").showModal();
  });
  $("helpClose").addEventListener("click", () => $("helpDialog").close());
  svg.addEventListener("click", (event) => {
    // SVG's matrix accounts for scaling and any letterboxing.
    const matrix = svg.getScreenCTM();
    if (!matrix) return;
    const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(
      matrix.inverse(),
    );
    const square = BR.pixelToBoard(point.x, point.y, opts());
    if (square) chooseSquare(square.x, square.y);
  });
  svg.addEventListener("focus", () => {
    keyboardFocus = true;
    draw();
  });
  svg.addEventListener("blur", () => {
    keyboardFocus = false;
    draw();
  });
  svg.addEventListener("keydown", (event) => {
    if (
      ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)
    ) {
      event.preventDefault();
      const dx = { ArrowLeft: -1, ArrowRight: 1 }[event.key] || 0;
      const dy = { ArrowUp: 1, ArrowDown: -1 }[event.key] || 0;
      cursor = {
        x: Math.max(0, Math.min(8, cursor.x + (flipped ? -dx : dx))),
        y: Math.max(0, Math.min(9, cursor.y + (flipped ? -dy : dy))),
      };
      draw();
      const ch = BR.parseAsciiBoard(boardAscii())[cursor.y][cursor.x];
      $("liveStatus").textContent = t("keyboardSquare", {
        square: "abcdefghi"[cursor.x] + (cursor.y + 1),
        piece: ch ? I.pieceName(ch) : t("empty"),
      });
    } else if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      chooseSquare(cursor.x, cursor.y);
    } else if (event.key === "Escape") {
      selected = null;
      draw();
      renderHelp();
    }
  });
  window
    .matchMedia("(min-width: 1000px)")
    .addEventListener("change", (event) => {
      if (event.matches && setupCollapsed) {
        setupCollapsed = false;
        render();
      }
    });
  document.addEventListener("languagechange", () => {
    if (catalog) buildControls();
    render();
  });
  initialize();
})();
