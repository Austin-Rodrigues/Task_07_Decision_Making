# Stakeholder Report
**Syracuse Women’s Lacrosse 2024 Season: Data-Driven Insights and Recommendations**

---

## Executive Summary
This report takes a fresh look at our **2024 season (16–6 record)** using official team statistics and new AI-supported analysis tools. The goal is simple: provide **clear, actionable steps** that can help us move closer to an **18–4 season** while keeping recommendations ethical, reliable, and easy to audit.

**What we learned:**
- Our top scorer, **Meaghan Tyrrell (70 goals, 61% shooting)**, led the attack with remarkable efficiency.
- Depth is strong: **9 players scored at least 10 goals**, showing multiple threats on offense.
- Shooting efficiency is high overall, but some mid-tier scorers (e.g., Rowley, Madnick) have room for improvement.
- Defensive and possession statistics are limited in the dataset, which makes it harder to fully balance recommendations.

**Our recommendations (tiered by risk):**
- **Low risk (operational):** Focus shooting drills on mid-efficiency scorers. Continue to use AI tools to summarize stats, but always cross-check with ground-truth numbers.
- **Medium risk (investigatory):** Collect richer defensive and possession data (turnovers, clears, draw control %) so that future analysis captures the whole field, not just scoring.
- **High risk (caution):** Do not rely on AI-generated insights alone for roster or recruiting decisions. Always involve coaching judgment and compliance review before acting on these.

**Confidence:** For factual numbers (totals, percentages), our confidence is high. For strategic recommendations, confidence is moderate and depends heavily on how the questions are asked.

---

## Background & Decision Context
- **Who this report is for:** Head Coach and Athletic Director  
- **Decision at hand:** Where to focus training and strategic attention to move from a **16–6 season** to an **18–4 season**  
- **What’s at stake:** Medium risk. Decisions shape training priorities, player development, and recruiting focus  

---

## Data & Methods
- **Data source:** Official Syracuse Women’s Lacrosse 2024 statistics (publicly available)  
- **How it was analyzed:**
  - A Python validation script recreated all basic totals (wins, goals, assists).
  - Shooting efficiency was tested with confidence ranges to check reliability.
  - AI (Large Language Models) was asked strategic questions. Their outputs were logged, then checked against the data.  
- **Transparency:** All prompts, responses, and ground-truth calculations are saved in the repository for audit. See `outputs/llm_responses_log.txt` for raw transcripts and validation outcomes.

---

## Findings

1. **Reliable facts**
   - Record: **16–6 across 22 games**
   - Team totals: **319 goals, 167 assists**
   - Top scorer: **Meaghan Tyrrell (70 goals, 61% shooting, CI 52–69%)**
   - Other standouts: Adamson (58G, 53%), Ward (44G, 49%)

2. **Depth on offense**
   - **9 players scored at least 10 goals**, confirming multiple contributors beyond stars.

3. **Room for growth**
   - Players like **Rowley (42% shooting)** and **Madnick (33%)** show opportunities to improve shot efficiency.

4. **Where AI struggled**
   - On “who scored ≥10 goals,” AI sometimes undercounted (8 instead of 9) unless given strict instructions.  
     *Evidence: see `outputs/llm_responses_log.txt` where the model miscounted and was corrected by stricter prompts.*  
   - Strategic suggestions tended to emphasize offense and overlook defense unless prompted directly.

---

## Recommendations

### Operational (Low Risk)
- Sharpen finishing drills for Rowley, Madnick, and similar mid-tier shooters.  
- Use AI tools for quick stat summaries, but always verify with team stats.

### Investigatory (Medium Risk)
- Start tracking defensive clears, turnovers, and draw control % in detail.  
- Use multiple AI tools and compare results to reduce reliance on a single output.

### High Stakes (High Risk)
- Recruiting and roster decisions should not rely on AI outputs alone.  
- Ensure any AI-based insights go through coach validation + compliance review.

---

## Ethical and Reliability Considerations
- **Bias:** AI tends to highlight offensive stats more than defensive ones. We balanced this by checking outputs against data and calling out missing context.  
- **Reliability:** Confidence intervals (CIs) were used to show how “solid” shooting percentages really are.  
- **Transparency:** Every AI prompt and output has been archived in `prompts/` and `outputs/`. Raw transcripts and validation outcomes are in `outputs/llm_responses_log.txt`.  
- **Privacy:** Only public statistics were used; no sensitive data involved.  

---

## Next Steps
1. Re-run the strict ≥10 goals prompt with multiple AI tools (Claude, ChatGPT, Copilot) and compare outputs.  
2. Collect richer defensive and possession data for a fuller picture of team performance.  
3. Review findings with coaching staff to align training drills with identified efficiency gaps.  

---

## Appendices
- Prompts & logs: `prompts/threshold_prompts_strict.txt`, `outputs/llm_responses_log.txt`  
- Ground truth stats: `outputs/ground_truth.json`  
- Fairness check: `outputs/fairness_usage.json`  
- Robustness/sensitivity tests: `outputs/robustness.json`, `outputs/sensitivity.json`  
- Sanity checks: `outputs/sanity_checks.json`  
