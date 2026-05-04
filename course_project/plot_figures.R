#!/usr/bin/env Rscript
# =============================================================================
#  Hybrid Chess — Publication-Quality Experiment Figures
#  Generates 6 PDF figures for the course report.
#  Usage:  Rscript course_project/plot_figures.R
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(scales)
  library(patchwork)
  library(RColorBrewer)
})

# ---------- paths -----------------------------------------------------------
runs_dir   <- "runs"
out_dir    <- "course_project/figures"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ---------- variant definitions ---------------------------------------------
variants <- tibble::tribble(
  ~label,         ~dir,                          ~short,
  "Default",      "rq4_az_default_v2",           "Def",
  "Q only",       "rq4_az_noq_only",             "Q",
  "X only",       "rq4_az_xqqueen_only",         "X",
  "PK",           "rq4_az_palace_knight_v2",      "PK",
  "PK+noPromo",   "rq4_az_pk_nopromo",            "PK+nP",
  "PK+xqQueen",   "rq4_az_pk_xqqueen",            "PK+xQ",
  "noQ+noPromo",  "rq4_az_nq_nopromo",            "nQ+nP",
  "noQ+PK",       "rq4_az_nq_pk",                 "nQ+PK",
  "noQ+ALL",      "rq4_az_nq_allrules_v2",        "nQ+ALL"
)

# ---------- load all metrics ------------------------------------------------
all_data <- purrr::map_dfr(seq_len(nrow(variants)), function(i) {
  path <- file.path(runs_dir, variants$dir[i], "metrics.csv")
  if (!file.exists(path)) return(NULL)
  d <- read.csv(path)
  d$variant <- variants$label[i]
  d$short   <- variants$short[i]
  d
})

# ---------- colour palette --------------------------------------------------
# Distinct, colour-blind friendly palette for 9 variants
variant_levels <- variants$label
variant_colours <- c(
  "Default"      = "#2C3E50",   # dark blue-grey
  "Q only"       = "#E74C3C",   # red
  "X only"       = "#27AE60",   # green  ⭐
  "PK"           = "#3498DB",   # blue
  "PK+noPromo"   = "#85C1E9",   # light blue
  "PK+xqQueen"   = "#1ABC9C",   # teal   ⭐
  "noQ+noPromo"  = "#F39C12",   # orange
  "noQ+PK"       = "#E67E22",   # dark orange
  "noQ+ALL"      = "#9B59B6"    # purple
)

# Common theme
theme_report <- theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(colour = "grey40", size = 10),
    legend.position = "right",
    panel.grid.minor = element_blank(),
    plot.margin = margin(10, 15, 10, 10)
  )

# =============================================================================
#  FIGURE 1: C:X Ratio Trends (THE core figure)
# =============================================================================
cat("Generating Fig 1: C:X Ratio Trends...\n")

cx_data <- all_data %>%
  mutate(
    cx_ratio = ifelse(sp_xiangqi_wins > 0,
                      sp_chess_wins / sp_xiangqi_wins,
                      NA_real_),
    variant = factor(variant, levels = variant_levels)
  )

# Rolling average (window=5) for smoother lines
cx_smooth <- cx_data %>%
  group_by(variant) %>%
  arrange(iter) %>%
  mutate(cx_smooth = zoo::rollmean(cx_ratio, k = 5, fill = NA, align = "center")) %>%
  ungroup()

p1 <- ggplot(cx_smooth, aes(x = iter, y = cx_smooth, colour = variant)) +
  geom_line(linewidth = 0.9, alpha = 0.85) +
  geom_hline(yintercept = 1, linetype = "dashed", colour = "grey50", linewidth = 0.5) +
  annotate("text", x = 48, y = 1.15, label = "Perfect Balance (1:1)",
           size = 3, colour = "grey50", hjust = 1) +
  scale_colour_manual(values = variant_colours, name = "Variant") +
  scale_y_continuous(limits = c(0, NA), breaks = seq(0, 20, 2)) +
  labs(
    title = "Chess:Xiangqi Win Ratio Across Training",
    subtitle = "Per-iteration C:X ratio (5-iter rolling avg) · 9 rule variants · 50 iters × 100 games",
    x = "Training Iteration",
    y = "C:X Win Ratio"
  ) +
  theme_report

ggsave(file.path(out_dir, "fig1_cx_ratio_trends.pdf"), p1,
       width = 10, height = 6, dpi = 300)
ggsave(file.path(out_dir, "fig1_cx_ratio_trends.png"), p1,
       width = 10, height = 6, dpi = 300)


# =============================================================================
#  FIGURE 2: Outcome Distribution (stacked bar)
# =============================================================================
cat("Generating Fig 2: Outcome Distribution...\n")

outcome_data <- all_data %>%
  group_by(variant) %>%
  summarise(
    Chess = sum(sp_chess_wins),
    Xiangqi = sum(sp_xiangqi_wins),
    Draw = sum(sp_draws),
    .groups = "drop"
  ) %>%
  mutate(Total = Chess + Xiangqi + Draw) %>%
  mutate(across(c(Chess, Xiangqi, Draw), ~ . / Total * 100)) %>%
  select(-Total) %>%
  pivot_longer(-variant, names_to = "outcome", values_to = "pct") %>%
  mutate(
    variant = factor(variant, levels = rev(variant_levels)),
    outcome = factor(outcome, levels = c("Chess", "Draw", "Xiangqi"))
  )

p2 <- ggplot(outcome_data, aes(x = variant, y = pct, fill = outcome)) +
  geom_col(position = "stack", width = 0.7) +
  coord_flip() +
  scale_fill_manual(
    values = c("Chess" = "#2C3E50", "Draw" = "#BDC3C7", "Xiangqi" = "#C0392B"),
    name = "Outcome"
  ) +
  labs(
    title = "Self-Play Outcome Distribution by Variant",
    subtitle = "Aggregated across 50 iterations (5,000 games per variant)",
    x = NULL,
    y = "Percentage (%)"
  ) +
  theme_report +
  theme(legend.position = "top")

ggsave(file.path(out_dir, "fig2_outcome_distribution.pdf"), p2,
       width = 9, height = 5.5, dpi = 300)
ggsave(file.path(out_dir, "fig2_outcome_distribution.png"), p2,
       width = 9, height = 5.5, dpi = 300)


# =============================================================================
#  FIGURE 3: Training Curves (Default variant — loss + eval)
# =============================================================================
cat("Generating Fig 3: Training Curves...\n")

train_data <- all_data %>% filter(variant == "Default")

# Panel A: loss curves
pa <- ggplot(train_data, aes(x = iter)) +
  geom_line(aes(y = policy_loss, colour = "Policy Loss"), linewidth = 0.8) +
  geom_line(aes(y = value_loss, colour = "Value Loss"), linewidth = 0.8) +
  scale_colour_manual(
    values = c("Policy Loss" = "#2980B9", "Value Loss" = "#E74C3C"),
    name = NULL
  ) +
  labs(title = "Training Loss", x = "Iteration", y = "Loss") +
  theme_report +
  theme(legend.position = c(0.8, 0.85))

# Panel B: eval win rate vs Random (where available)
eval_data <- train_data %>%
  filter(!is.na(eval_random_w)) %>%
  mutate(
    total_rand = eval_random_w + eval_random_d + eval_random_l,
    wr_random = ifelse(total_rand > 0, eval_random_w / total_rand * 100, NA),
    total_ab = eval_ab_w + eval_ab_d + eval_ab_l,
    wr_ab = ifelse(total_ab > 0, eval_ab_w / total_ab * 100, NA)
  )

pb <- ggplot(eval_data, aes(x = iter)) +
  geom_line(aes(y = wr_random, colour = "vs Random"), linewidth = 0.8) +
  geom_line(aes(y = wr_ab, colour = "vs AB(d1)"), linewidth = 0.8) +
  scale_colour_manual(
    values = c("vs Random" = "#27AE60", "vs AB(d1)" = "#8E44AD"),
    name = NULL
  ) +
  scale_y_continuous(limits = c(0, 100)) +
  labs(title = "Evaluation Win Rate", x = "Iteration", y = "Win Rate (%)") +
  theme_report +
  theme(legend.position = c(0.75, 0.2))

p3 <- pa + pb +
  plot_annotation(
    title = "AlphaZero Training Convergence (Default Variant)",
    subtitle = "50 iterations × 100 self-play games · eval every 2 iters",
    theme = theme(plot.title = element_text(face = "bold", size = 14),
                  plot.subtitle = element_text(colour = "grey40", size = 10))
  )

ggsave(file.path(out_dir, "fig3_training_curves.pdf"), p3,
       width = 11, height = 5, dpi = 300)
ggsave(file.path(out_dir, "fig3_training_curves.png"), p3,
       width = 11, height = 5, dpi = 300)


# =============================================================================
#  FIGURE 4: Factor Analysis Heatmap (Queen × PK → C:X & Draw%)
# =============================================================================
cat("Generating Fig 4: Factor Heatmap...\n")

factor_data <- tibble::tribble(
  ~queen_config,      ~pk_config,    ~cx_ratio, ~draw_pct, ~label_cx, ~label_draw,
  "Chess Q / XQ no Q", "Without PK",  9.0,       67,       "9.0x",    "67%",
  "Chess Q / XQ no Q", "With PK",     3.4,       61,       "3.4x",    "61%",
  "Chess Q / XQ has Q", "Without PK", 0.7,       58,       "0.7x",    "58%",
  "Chess Q / XQ has Q", "With PK",    0.7,       54,       "0.7x",    "54%",
  "No Chess Q / XQ no Q", "Without PK", 4.5,     86,       "4.5x",    "86%",
  "No Chess Q / XQ no Q", "With PK",    0.6,     88,       "0.6x",    "88%"
) %>%
  mutate(
    queen_config = factor(queen_config,
      levels = c("Chess Q / XQ no Q", "Chess Q / XQ has Q", "No Chess Q / XQ no Q")),
    pk_config = factor(pk_config, levels = c("Without PK", "With PK")),
    cell_label = paste0("C:X = ", label_cx, "\nDraw = ", label_draw)
  )

p4 <- ggplot(factor_data, aes(x = pk_config, y = queen_config, fill = cx_ratio)) +
  geom_tile(colour = "white", linewidth = 1.5) +
  geom_text(aes(label = cell_label), size = 4, fontface = "bold") +
  scale_fill_gradient2(
    low = "#27AE60", mid = "#F9E79F", high = "#E74C3C",
    midpoint = 3, limits = c(0, 10),
    name = "C:X Ratio"
  ) +
  labs(
    title = "Factor Analysis: Queen Config × Structural Reform",
    subtitle = "C:X ratio and draw rate for each factor combination",
    x = "Structural Reform (PK = Palace + Knight Block)",
    y = "Queen Configuration"
  ) +
  theme_report +
  theme(
    panel.grid = element_blank(),
    axis.text = element_text(size = 11)
  )

ggsave(file.path(out_dir, "fig4_factor_heatmap.pdf"), p4,
       width = 9, height = 6, dpi = 300)
ggsave(file.path(out_dir, "fig4_factor_heatmap.png"), p4,
       width = 9, height = 6, dpi = 300)


# =============================================================================
#  FIGURE 5: Payoff Matrix Heatmap (Tournament)
# =============================================================================
cat("Generating Fig 5: Payoff Matrix...\n")

payoff_csv <- file.path(runs_dir, "cross_variant_tournament", "payoff_matrix.csv")
payoff <- read.csv(payoff_csv, row.names = 1, check.names = FALSE)

payoff_long <- payoff %>%
  tibble::rownames_to_column("row_agent") %>%
  pivot_longer(-row_agent, names_to = "col_agent", values_to = "score") %>%
  mutate(
    row_agent = factor(row_agent, levels = rev(rownames(payoff))),
    col_agent = factor(col_agent, levels = colnames(payoff)),
    label = sprintf("%.2f", score)
  )

p5 <- ggplot(payoff_long, aes(x = col_agent, y = row_agent, fill = score)) +
  geom_tile(colour = "white", linewidth = 0.8) +
  geom_text(aes(label = label), size = 3.2, fontface = "bold") +
  scale_fill_gradient2(
    low = "#2980B9", mid = "#ECF0F1", high = "#E74C3C",
    midpoint = 0.5, limits = c(0, 1),
    name = "Score"
  ) +
  labs(
    title = "Cross-Variant Tournament Payoff Matrix",
    subtitle = "9 agents × 50 games per pair (1,800 total) · played under Default rules",
    x = "Opponent",
    y = "Agent"
  ) +
  theme_report +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
    axis.text.y = element_text(size = 9),
    panel.grid = element_blank()
  )

ggsave(file.path(out_dir, "fig5_payoff_matrix.pdf"), p5,
       width = 9, height = 7, dpi = 300)
ggsave(file.path(out_dir, "fig5_payoff_matrix.png"), p5,
       width = 9, height = 7, dpi = 300)


# =============================================================================
#  FIGURE 6: Tournament Ranking (horizontal bar)
# =============================================================================
cat("Generating Fig 6: Tournament Ranking...\n")

ranking_data <- tibble::tribble(
  ~agent,        ~score,  ~training_rule,
  "Q only",      0.625,   "Remove Chess Queen",
  "PK",          0.625,   "Palace + Knight Block",
  "noQ+ALL",     0.562,   "All Restrictions",
  "X only",      0.531,   "Give XQ a Queen",
  "Default",     0.500,   "Standard Rules",
  "PK+noPromo",  0.469,   "PK + No Promotion",
  "noQ+noPromo", 0.406,   "noQ + No Promotion",
  "noQ+PK",      0.406,   "noQ + PK",
  "PK+xqQueen",  0.375,   "PK + XQ Queen"
) %>%
  mutate(agent = factor(agent, levels = rev(agent)))

p6 <- ggplot(ranking_data, aes(x = agent, y = score, fill = score)) +
  geom_col(width = 0.65) +
  geom_text(aes(label = sprintf("%.3f", score)),
            hjust = -0.15, size = 3.8, fontface = "bold") +
  geom_text(aes(label = training_rule, y = 0.01),
            hjust = 0, size = 3, colour = "white", fontface = "italic") +
  geom_hline(yintercept = 0.5, linetype = "dashed", colour = "grey50") +
  coord_flip(ylim = c(0, 0.72)) +
  scale_fill_gradient2(
    low = "#2980B9", mid = "#BDC3C7", high = "#E74C3C",
    midpoint = 0.5, guide = "none"
  ) +
  labs(
    title = "Cross-Variant Tournament: Agent Ranking",
    subtitle = "Avg score across 8 opponents · Agents trained under different rule variants",
    x = NULL,
    y = "Average Score"
  ) +
  theme_report

ggsave(file.path(out_dir, "fig6_tournament_ranking.pdf"), p6,
       width = 9, height = 5.5, dpi = 300)
ggsave(file.path(out_dir, "fig6_tournament_ranking.png"), p6,
       width = 9, height = 5.5, dpi = 300)


# =============================================================================
cat("\n✅ All 6 figures generated in:", out_dir, "\n")
cat("   fig1_cx_ratio_trends     — C:X ratio evolution (core figure)\n")
cat("   fig2_outcome_distribution — Win/Draw/Loss stacked bars\n")
cat("   fig3_training_curves      — Loss + eval convergence\n")
cat("   fig4_factor_heatmap       — Queen × PK interaction\n")
cat("   fig5_payoff_matrix        — 9×9 tournament heatmap\n")
cat("   fig6_tournament_ranking   — Agent ranking bar chart\n")
