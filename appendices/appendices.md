# Appendices – Prompts, Outputs, and Edit Log

---

## A. Prompts (exact text)
- Threshold (strict): see `prompts/threshold_prompts_strict.txt`
- Other prompts used for testing are documented in `outputs/llm_responses_log.txt`

---

## B. Raw LLM Outputs
- All raw transcripts and validation results are saved in `outputs/llm_responses_log.txt`.
- Examples include:
  - ✓ Correct answers for season record, total games, top scorer, and team totals.
  - ✗ Incorrect response for “≥10 goals list” (AI gave 8 instead of 9).
  - Mixed quality in strategic analysis responses (often offense-heavy).

---

## C. Edit Log (what changed & why)
- **Threshold miscount:** Corrected ≥10-goal count to **9** based on `outputs/ground_truth.json`.  
- **Strategic balance:** Human edits emphasized defensive/possession metrics to offset AI offense-only focus.  
- **Formatting:** Smoothed technical language into coach-friendly recommendations.  

---

## D. Reproducibility Notes
- **Script:** `scripts/Testing_and_Validation.py`  
- **Python:** 3.13  
- **Workflow:** Run script → launcher option `2) Export ALL Task 07 artifacts` → outputs populate under `/outputs/`.  
- **Validation evidence:** See `outputs/llm_responses_log.txt` for full transcripts and checks.  
