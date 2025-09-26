# Task 07 – Decision Making with LLMs (Syracuse Women’s Lacrosse 2024)

This repository converts prior LLM experiments into a **stakeholder-facing, auditable** decision report that incorporates uncertainty, robustness, and ethical safeguards.

---

## Structure
- `report/` – Final stakeholder-facing report (`Stakeholder_Report.pdf`, `Stakeholder_Report.md`)
- `appendices/` – Supporting files (`Appendices.md`, raw LLM outputs, edit notes)
- `data/` – Context file (`syracuse_data_context.txt`) — no raw dataset included
- `scripts/` – Python validation script (`Testing_and_Validation.py`)
- `prompts/` – Test prompts (`threshold_prompts_strict.txt`)
- `outputs/` – Generated artifacts:
  - `ground_truth.json`
  - `fairness_usage.json`
  - `robustness.json`
  - `sensitivity.json`
  - `sanity_checks.json`
  - `validation_results.json`
  - `syracuse_testing_report.md`
  - `llm_responses_log.txt`

---

## How to Reproduce
1. Run `scripts/Testing_and_Validation.py`.
2. From the launcher, choose **2) Export ALL Task 07 artifacts**.
3. Review `outputs/*.json` and the report in `report/Stakeholder_Report.pdf`.

---

## Key Takeaways
- LLMs are reliable for **facts and simple calculations**, but fragile for **threshold lists** and **generic strategy** unless tightly prompted.
- We quantify **uncertainty** (Wilson 95% confidence intervals), check **fairness proxies**, and test **robustness/sensitivity**.
- LLM outputs are **advisory only**; human + compliance review is required for high-stakes decisions.

---

## Contact
Austin Anthony Rodrigues  
Instructor: Dr. J. R. Strome (jrstrome@syr.edu)
