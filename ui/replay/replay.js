(() => {
  const I = window.HybridI18n,
    BR = window.BoardRenderer,
    F = window.HybridReplayFormat,
    $ = (id) => document.getElementById(id),
    t = I.t;
  let games = [],
    gameIndex = 0,
    step = 0,
    timer = null,
    flipped = false,
    initial = "",
    errorCode = "",
    fileSequence = 0;
  const game = () => games[gameIndex],
    total = () => Math.max(0, (game()?.states_ascii.length || 1) - 1);
  function element(tag, className = "", value = "") {
    const el = document.createElement(tag);
    el.className = className;
    el.textContent = value;
    return el;
  }
  function variant() {
    const v = game()?.variant || game()?.meta?.variant;
    return v && typeof v === "object" && !Array.isArray(v) ? v : {};
  }
  function resultLabel(g) {
    const code = g?.result_code || g?.result;
    if (
      ["chess_win", "xiangqi_win", "draw", "ongoing"].includes(code)
    )
      return t(code);
    const value = g?.result || "";
    if (/^Chess wins/i.test(value)) return t("chess_win");
    if (/^Xiangqi wins/i.test(value)) return t("xiangqi_win");
    if (/^(Draw|1\/2-1\/2)$/i.test(value)) return t("draw");
    return value || "—";
  }
  function reasonLabel(g) {
    const key =
      g?.reason_code ||
      {
        Checkmate: "checkmate",
        "Stalemate (loss for stalemated side)": "stalemate",
        "Max plies reached": "move_limit",
        "Threefold repetition": "repetition",
        "Chess king captured": "royal_captured",
        "Xiangqi general captured": "royal_captured",
        Resignation: "resignation",
      }[g?.meta?.reason];
    return key ? t(key) : g?.meta?.reason || "";
  }
  function render() {
    I.apply();
    document.title = "Hybrid Chess · " + t("replay");
    $("errorNotice").hidden = !errorCode;
    $("errorText").textContent = errorCode ? t(errorCode) : "";
    const o = { ...BR.defaultOpts(), flipped, variant: variant() };
    BR.drawBoard($("boardSvg"), o);
    BR.drawCoordLabels($("boardSvg"), o);
    BR.clearHighlights($("boardSvg"));
    BR.drawPieces($("boardSvg"), game()?.states_ascii[step] || initial, o);
    const last = step > 0 ? F.parseMove(game()?.moves?.[step - 1]) : null;
    if (last) BR.highlightMove($("boardSvg"), last.from, last.to, o);
    $("stepLabel").textContent = t("position", { step, total: total() });
    $("boardSvg").setAttribute(
      "aria-label",
      t("replay") + " · " + t("position", { step, total: total() }),
    );
    $("boardCaption").textContent = t(games.length ? "loaded" : "preview");
    $("stepSlider").max = total();
    $("stepSlider").value = step;
    $("stepSlider").disabled = !games.length;
    $("btnFirst").disabled = $("btnPrev").disabled =
      !games.length || step === 0;
    $("btnNext").disabled = $("btnLast").disabled =
      !games.length || step >= total();
    $("btnAuto").disabled = !games.length || total() === 0;
    $("btnAuto").textContent = timer ? "Ⅱ" : "▷";
    $("btnAuto").setAttribute("aria-label", t(timer ? "pause" : "autoPlay"));
    $("btnAuto").title = t(timer ? "pause" : "autoPlay");
    $("infoResult").textContent = resultLabel(game());
    $("infoReason").textContent = reasonLabel(game());
    $("infoVariant").textContent =
      games.length && !Object.keys(variant()).length ? t("legacyRules") : "";
    $("gameSelectGroup").hidden = games.length < 2;
    $("gameSelect").replaceChildren(
      ...games.map((g, i) => {
        const option = element(
          "option",
          "",
          t("gameNumber", { number: i + 1 }) + " · " + resultLabel(g),
        );
        option.value = i;
        return option;
      }),
    );
    $("gameSelect").value = gameIndex;
    $("moveList").replaceChildren();
    const moves = game()?.moves || [];
    if (!moves.length)
      $("moveList").append(
        element(
          "p",
          "empty-recording",
          t(games.length ? "noRecordedMoves" : "replayEmpty"),
        ),
      );
    moves.forEach((move, index) => {
      const button = element(
        "button",
        "replay-move" + (index === step - 1 ? " active" : ""),
      );
      button.append(
        element("span", "number", String(index + 1) + "."),
        element("span", "", move),
      );
      button.disabled = index >= total();
      button.setAttribute(
        "aria-current",
        index === step - 1 ? "step" : "false",
      );
      button.addEventListener("click", () => go(index + 1));
      $("moveList").append(button);
    });
    const live = t("position", { step, total: total() });
    if ($("replayLive").textContent !== live)
      $("replayLive").textContent = live;
  }
  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
  }
  function go(value, manual = true) {
    if (manual) stop();
    step = Math.max(0, Math.min(total(), value));
    render();
    $("moveList")
      .querySelector(".active")
      ?.scrollIntoView({ block: "nearest" });
  }
  function toggle() {
    if (timer) {
      stop();
      render();
      return;
    }
    if (!games.length || !total()) return;
    if (step >= total()) step = 0;
    timer = setInterval(() => {
      step = Math.min(total(), step + 1);
      if (step >= total()) stop();
      render();
    }, 700);
    render();
  }
  $("btnFirst").addEventListener("click", () => go(0));
  $("btnPrev").addEventListener("click", () => go(step - 1));
  $("btnNext").addEventListener("click", () => go(step + 1));
  $("btnLast").addEventListener("click", () => go(total()));
  $("btnAuto").addEventListener("click", toggle);
  $("btnFlip").addEventListener("click", () => {
    flipped = !flipped;
    render();
  });
  $("stepSlider").addEventListener("input", (event) =>
    go(Number(event.target.value)),
  );
  $("gameSelect").addEventListener("change", (event) => {
    stop();
    gameIndex = Number(event.target.value);
    step = 0;
    render();
  });
  $("btnOpenReplay").addEventListener("click", () => $("fileInput").click());
  $("fileInput").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    stop();
    const sequence = ++fileSequence;
    try {
      if (file.size > 20 * 1024 * 1024) throw new Error("fileTooLarge");
      const parsed = F.parse(await file.text());
      if (sequence !== fileSequence) return;
      games = parsed;
      gameIndex = 0;
      step = 0;
      errorCode = "";
      $("fileName").textContent = file.name;
    } catch (error) {
      if (sequence !== fileSequence) return;
      errorCode =
        error.message === "fileTooLarge" ? "fileTooLarge" : "replayBad";
    }
    event.target.value = "";
    render();
  });
  document.addEventListener("keydown", (event) => {
    if (event.target.closest("input,select,button,a,summary") || !games.length)
      return;
    const actions = {
      ArrowLeft: () => go(step - 1),
      ArrowRight: () => go(step + 1),
      Home: () => go(0),
      End: () => go(total()),
      " ": toggle,
    };
    if (actions[event.key]) {
      event.preventDefault();
      actions[event.key]();
    }
  });
  document.addEventListener("languagechange", render);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      stop();
      render();
    }
  });
  render();
  fetch("/api/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ variant: "none" }),
  })
    .then((r) => (r.ok ? r.json() : null))
    .then((data) => {
      initial = data?.board_ascii || "";
      if (!games.length) render();
    })
    .catch(() => {});
})();
