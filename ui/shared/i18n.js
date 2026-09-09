/* Shared copy and local preferences for play and replay. */
(() => {
  const copy = {
    play: ["对弈", "Play"],
    replay: ["回放", "Replays"],
    rulesHelp: ["规则指南", "Rule guide"],
    tagline: ["两种棋艺，一张棋盘。", "Two traditions. One board."],
    playTitle: ["选择规则，落子开局。", "Choose your rules. Make your move."],
    playSubtitle: ["国际象棋 × 中国象棋 · 9 × 10", "Chess × Xiangqi · 9 × 10"],
    language: ["切换为英文", "Switch to Chinese"],
    board: [
      "棋盘：方向键移动，回车选子或落子，Escape 取消",
      "Board: arrow keys to navigate, Enter to select or move, Escape to cancel",
    ],
    setup: ["新对局设置", "Set up your game"],
    preset: ["选择规则", "Choose your rules"],
    custom: ["自定义规则", "Customize rules"],
    customName: ["自定义组合", "Custom rules"],
    customDesc: [
      "组合现有规则，试试不同的对弈方式。",
      "Combine rules and explore a different game.",
    ],
    group_chess: ["国际象棋棋子", "Chess pieces"],
    group_xiangqi: ["中国象棋棋子", "Xiangqi pieces"],
    group_movement: ["行棋规则", "Movement rules"],
    yourSide: ["选择阵营", "Choose your side"],
    chess: ["国际象棋", "Chess"],
    xiangqi: ["中国象棋", "Xiangqi"],
    chessShort: ["国际象棋", "Chess"],
    xiangqiShort: ["中国象棋", "Xiangqi"],
    chessFirst: [
      "国际象棋先行。选择象棋时，AI 会先走。",
      "Chess moves first. Choose Xiangqi and the AI opens.",
    ],
    opponent: ["AI 对手", "AI opponent"],
    thinkBudget: [
      "每步约 {seconds} 秒，完成深度取决于局面。",
      "About {seconds}s per move; search depth depends on the position.",
    ],
    practiceHint: [
      "适合熟悉棋子走法与规则。",
      "A relaxed way to learn the pieces and rules.",
    ],
    start: ["开始对弈", "Start playing"],
    newGame: ["新对局", "New game"],
    cancelSetup: ["返回当前对局", "Back to game"],
    ready: ["等待开局", "Ready to play"],
    preview: ["开局预览", "Starting position"],
    previewLoading: ["更新预览…", "Updating preview…"],
    loading: ["连接棋桌…", "Connecting…"],
    you: ["你", "You"],
    ai: ["AI", "AI"],
    yourTurn: ["轮到你了", "Your move"],
    aiTurn: ["等待 AI 落子", "Waiting for AI"],
    thinking: ["AI 正在思考…", "AI is thinking…"],
    sending: ["正在落子…", "Making move…"],
    check: ["将军", "Check"],
    turnCount: ["第 {count} 步", "Ply {count}"],
    boardHint: [
      "点击棋子查看走法，再点击落点。",
      "Select a piece, then choose a highlighted square.",
    ],
    beforeStart: [
      "选好规则，开始你的第一局。",
      "Choose your rules and start a game.",
    ],
    moveHint: [
      "实心点为可走位置，圆环为可吃的棋子。",
      "Dots show moves. Rings show captures.",
    ],
    flip: ["翻转棋盘", "Flip board"],
    undo: ["悔棋", "Undo"],
    resign: ["认输", "Resign"],
    currentRules: ["本局规则", "This game’s rules"],
    ruleCount: ["{count} 项改动", "{count} changes"],
    unchanged: [
      "使用标准走法与阵容。",
      "Original movement rules and starting pieces.",
    ],
    disabledFlying: ["关闭飞将", "Flying general disabled"],
    moves: ["棋谱", "Moves"],
    noMoves: ["第一步，由你开始。", "The first move is yours."],
    noMovesAI: ["AI 将执国际象棋先行。", "The AI opens with Chess."],
    selectedPiece: ["已选择 {piece} · {square}", "{piece} selected · {square}"],
    availableMoves: ["{count} 个可走位置", "{count} available squares"],
    gameOver: ["对局结束", "Game over"],
    chess_win: ["国际象棋获胜", "Chess wins"],
    xiangqi_win: ["中国象棋获胜", "Xiangqi wins"],
    draw: ["和棋", "Draw"],
    ongoing: ["对弈中", "In progress"],
    checkmate: ["将死", "Checkmate"],
    stalemate: ["无棋可走，判负", "Stalemate is a loss"],
    move_limit: ["达到步数上限", "Move limit reached"],
    repetition: ["三次重复局面", "Threefold repetition"],
    royal_captured: ["王或将被吃", "Royal captured"],
    resignation: ["认输结束", "Resignation"],
    again: ["再来一局", "Play again"],
    changeRules: ["更换规则", "Change rules"],
    confirmNew: ["重新开局？", "Start a new game?"],
    confirmNewBody: [
      "当前对局会被替换。新规则只在新对局中生效。",
      "This replaces the current game. The new rules apply to the new game only.",
    ],
    confirmResign: ["确认认输？", "Resign this game?"],
    confirmResignBody: [
      "本局将结束，AI 获胜。",
      "This ends the game with a win for the AI.",
    ],
    confirm: ["确认", "Confirm"],
    cancel: ["取消", "Cancel"],
    close: ["关闭", "Close"],
    promotion: ["选择升变棋子", "Choose a promotion"],
    promotionBody: [
      "这些是本局规则允许的升变。",
      "These promotions are legal under the current rules.",
    ],
    retry: ["重试", "Retry"],
    reconnect: ["重新连接", "Reconnect"],
    sync: ["同步对局", "Sync game"],
    network: [
      "连接中断。请确认本地服务仍在运行，再重新连接。",
      "Connection lost. Check that the local server is running, then reconnect.",
    ],
    no_game: [
      "没有进行中的对局，请重新开局。",
      "There is no active game. Start a new one.",
    ],
    invalid_variant: [
      "规则配置无效，请重新选择。",
      "Invalid rules. Please choose them again.",
    ],
    invalid_side: ["请选择有效阵营。", "Please choose a valid side."],
    invalid_agent: ["请选择有效 AI。", "Please choose a valid AI."],
    invalid_move: ["走法格式无效。", "Invalid move."],
    invalid_promotion: ["不能升变为该棋子。", "That promotion is not allowed."],
    illegal_move: ["这一步不符合本局规则。", "That move is not legal."],
    wrong_turn: [
      "当前不是你的回合，已同步棋局。",
      "The turn has changed. The game has been synchronized.",
    ],
    stale_state: [
      "棋局已更新，已同步到最新状态。",
      "The game changed. The latest position has been synchronized.",
    ],
    game_over: ["对局已结束。", "This game has ended."],
    nothing_to_undo: [
      "还没有可以撤回的走棋。",
      "There is no player move to undo.",
    ],
    invalid_request: [
      "请求无效，请同步后重试。",
      "Invalid request. Sync the game and try again.",
    ],
    server_error: [
      "处理失败。请重试或重新开局。",
      "Something went wrong. Retry or start a new game.",
    ],
    not_found: [
      "未找到此功能，请刷新页面。",
      "Not found. Please refresh the page.",
    ],
    guideIntro: [
      "国际象棋在下，中国象棋在上。双方在交叉点上落子，国际象棋先行。",
      "Chess starts below and Xiangqi above. Pieces occupy intersections; Chess moves first.",
    ],
    guideEnding: [
      "将死、吃掉王或将，或者让对手无棋可走即可获胜。三次重复局面或达到 400 步判和。没有王车易位或吃过路兵。",
      "Win by checkmate, capturing the royal, or leaving the opponent without a legal move. Threefold repetition or 400 plies is a draw. No castling or en passant.",
    ],
    guideVariant: [
      "选子时显示本局走法。自定义规则会改变部分棋子的移动方式和初始阵容。",
      "Select a piece to see its current movement rules. Custom rules can change movement and the starting army.",
    ],
    king: ["王", "King"],
    queen: ["皇后", "Queen"],
    rook: ["车", "Rook"],
    bishop: ["象", "Bishop"],
    knight: ["马", "Knight"],
    pawn: ["兵", "Pawn"],
    general: ["将", "General"],
    advisor: ["士", "Advisor"],
    elephant: ["象", "Elephant"],
    horse: ["马", "Horse"],
    chariot: ["车", "Chariot"],
    cannon: ["炮", "Cannon"],
    soldier: ["卒", "Soldier"],
    xqQueen: ["象棋皇后", "Xiangqi queen"],
    kingMove: ["沿任意方向走一格。", "One square in any direction."],
    kingPalace: [
      "沿任意方向走一格，限于己方九宫。",
      "One square in any direction, inside the Chess palace.",
    ],
    queenMove: [
      "沿直线或斜线走任意格，不能越子。",
      "Slide along ranks, files or diagonals; cannot jump.",
    ],
    rookMove: [
      "沿横线或竖线走任意格，不能越子。",
      "Slide along ranks or files; cannot jump.",
    ],
    bishopMove: [
      "沿斜线走任意格，不能越子。",
      "Slide diagonally; cannot jump.",
    ],
    knightMove: [
      "走日字，可以跳过棋子。",
      "Move in an L shape; can jump over pieces.",
    ],
    horseMove: [
      "走日字，马腿处有棋子时不能走。",
      "Move in an L shape; a piece at the leg blocks the move.",
    ],
    pawnMove: [
      "向前一格，初始可走两格；斜前方吃子。到底线可升变。",
      "Forward one, or two from its starting rank; capture diagonally. Promote on the last rank.",
    ],
    pawnNoPromo: [
      "向前一格，初始可走两格；斜前方吃子。不能升变。",
      "Forward one, or two from its starting rank; capture diagonally. No promotion.",
    ],
    pawnNoQueen: [
      "向前走、斜前方吃子；到底线仅可升为车、象或马。",
      "Move forward and capture diagonally. Promote only to rook, bishop or knight.",
    ],
    generalMove: [
      "在九宫内沿横线或竖线走一格。",
      "One orthogonal square inside the palace.",
    ],
    flyingMove: [
      "同列无子阻隔时，也可飞将吃王。",
      "Can also capture the king along a clear file.",
    ],
    advisorMove: [
      "在九宫内斜走一格。",
      "One diagonal square inside the palace.",
    ],
    elephantMove: [
      "斜走两格，不能塞象眼，也不能过河。",
      "Two diagonal squares; the eye must be clear. Cannot cross the river.",
    ],
    cannonMove: [
      "沿横线或竖线移动；吃子必须恰好隔一个棋子。",
      "Slide orthogonally. Captures jump exactly one intervening piece.",
    ],
    soldierMove: [
      "过河前向前一格；过河后也可横走，不能后退。",
      "Forward one square; after crossing the river, also sideways. Never backward.",
    ],
    keyboardSquare: ["{square}，{piece}", "{square}, {piece}"],
    empty: ["空位", "empty"],
    replayTitle: ["重温每一步。", "Every move, revisited."],
    replaySubtitle: [
      "载入棋谱，慢慢看清一局棋。",
      "Open a recording and explore the game at your pace.",
    ],
    openReplay: ["打开棋谱", "Open recording"],
    replayEmpty: ["打开 JSON 或 JSONL 棋谱", "Open a JSON or JSONL recording"],
    replayEmptyHelp: [
      "支持项目导出的单局或多局记录。",
      "Single-game and multi-game exports are supported.",
    ],
    first: ["回到开局", "First position"],
    previous: ["上一步", "Previous move"],
    next: ["下一步", "Next move"],
    last: ["最后一步", "Last position"],
    autoPlay: ["自动播放", "Auto-play"],
    pause: ["暂停", "Pause"],
    position: ["第 {step} / {total} 步", "Ply {step} / {total}"],
    gameSelect: ["选择对局", "Choose a game"],
    gameNumber: ["对局 {number}", "Game {number}"],
    replayBad: [
      "无法读取棋谱。需要包含 states_ascii 的有效 JSON 或 JSONL 记录。",
      "Cannot read this recording. Use valid JSON or JSONL with states_ascii positions.",
    ],
    loaded: ["棋谱已载入", "Recording loaded"],
    replayResult: ["记录结果", "Recorded result"],
    legacyRules: [
      "此记录未附带规则配置。",
      "This recording has no rule configuration.",
    ],
    noRecordedMoves: ["没有走子记录", "No recorded moves"],
    fileTooLarge: [
      "文件过大，请选择 20 MB 以内的棋谱。",
      "Please choose a recording smaller than 20 MB.",
    ],
    localOnly: ["本地对弈 · 无需账号", "Local play · No account needed"],
  };
  const preference = {
    get(key, fallback) {
      try {
        return JSON.parse(localStorage.getItem("hybrid." + key)) ?? fallback;
      } catch {
        return fallback;
      }
    },
    set(key, value) {
      try {
        localStorage.setItem("hybrid." + key, JSON.stringify(value));
      } catch {}
    },
  };
  let lang = preference.get("language", "zh");
  if (!["zh", "en"].includes(lang)) lang = "zh";
  function t(key, values = {}) {
    let result = copy[key]?.[lang === "zh" ? 0 : 1] ?? key;
    for (const [name, value] of Object.entries(values))
      result = result.replaceAll("{" + name + "}", String(value));
    return result;
  }
  function localized(value) {
    return value?.[lang] ?? value?.en ?? "";
  }
  function apply() {
    document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
    document.querySelectorAll("[data-i18n]").forEach((el) => {
      el.textContent = t(el.dataset.i18n);
    });
    document.querySelectorAll("[data-i18n-label]").forEach((el) => {
      el.setAttribute("aria-label", t(el.dataset.i18nLabel));
      el.title = t(el.dataset.i18nLabel);
    });
    document.querySelectorAll("[data-language]").forEach((el) => {
      el.textContent = lang === "zh" ? "EN" : "中文";
      el.setAttribute("aria-label", t("language"));
    });
  }
  function setLanguage(value) {
    lang = value;
    preference.set("language", value);
    apply();
    document.dispatchEvent(new CustomEvent("languagechange"));
  }
  const kinds = {
    K: "king",
    Q: "queen",
    R: "rook",
    B: "bishop",
    N: "knight",
    P: "pawn",
    g: "general",
    a: "advisor",
    e: "elephant",
    h: "horse",
    c: "chariot",
    n: "cannon",
    s: "soldier",
    q: "xqQueen",
  };
  function pieceDescription(ch, v = {}) {
    const key = {
      K: v.chess_palace ? "kingPalace" : "kingMove",
      Q: "queenMove",
      R: "rookMove",
      B: "bishopMove",
      N: v.knight_block ? "horseMove" : "knightMove",
      P: v.no_promotion
        ? "pawnNoPromo"
        : v.no_queen_promotion
          ? "pawnNoQueen"
          : "pawnMove",
      g: "generalMove",
      a: "advisorMove",
      e: "elephantMove",
      h: "horseMove",
      c: "rookMove",
      n: "cannonMove",
      s: "soldierMove",
      q: "queenMove",
    }[ch];
    return (
      t(key) +
      (ch === "g" && v.flying_general !== false ? " " + t("flyingMove") : "")
    );
  }
  document.addEventListener("click", (e) => {
    if (e.target.closest("[data-language]"))
      setLanguage(lang === "zh" ? "en" : "zh");
  });
  window.HybridI18n = {
    t,
    localized,
    apply,
    preference,
    pieceDescription,
    pieceName: (ch) => t(kinds[ch] || "empty"),
    get lang() {
      return lang;
    },
  };
})();
