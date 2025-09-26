#!/usr/bin/env python3
"""
Task 07 Upgraded: Testing & Validation Framework
Syracuse Women's Lacrosse 2024 Statistics → LLM Evaluation with
Uncertainty, Fairness Proxy, Robustness/Sensitivity, and Full Archiving

Adds:
- Wilson CIs for shooting% (uncertainty)
- Usage-based fairness proxy (high vs low usage)
- Robustness (remove top-N scorers)
- Sensitivity (change threshold for ≥N goals)
- Sanity checks (data anomalies)
- Strict threshold prompts
- One-click export of all Task 07 outputs
"""

import os
import json
import logging
import re
from math import sqrt
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd


# ----------------------------- Utilities & IO ----------------------------- #

def ensure_dirs():
    """Create folder structure if missing."""
    for d in ["outputs", "prompts", "data"]:
        os.makedirs(d, exist_ok=True)


def save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def now_iso():
    return datetime.now().isoformat()


# ----------------------------- Results Analyzer ----------------------------- #

class ResultsAnalyzer:
    """Analyze and summarize LLM testing results"""

    def __init__(self):
        self.results = []

    def add_result(self, prompt_type: str, question: str, llm_response: str,
                   validation_result: Dict[str, Any]):
        """Add a test result"""
        self.results.append({
            'timestamp': now_iso(),
            'prompt_type': prompt_type,
            'question': question,
            'llm_response': llm_response,
            'validation': validation_result
        })

    def generate_summary_report(self) -> str:
        """Generate comprehensive summary report"""
        if not self.results:
            return "No results to analyze."

        # Calculate success rates by type
        type_stats = {}
        for result in self.results:
            ptype = result['prompt_type']
            if ptype not in type_stats:
                type_stats[ptype] = {'total': 0, 'accurate': 0}

            type_stats[ptype]['total'] += 1
            if result['validation'].get('accuracy', False):
                type_stats[ptype]['accurate'] += 1

        report = "# LLM Testing Summary Report\n\n"
        report += f"**Total Tests Conducted:** {len(self.results)}\n"
        report += f"**Test Date:** {datetime.now().strftime('%Y-%m-%d')}\n\n"

        report += "## Success Rates by Question Type\n\n"
        for ptype, stats in type_stats.items():
            success_rate = (stats['accurate'] / stats['total']
                            ) * 100 if stats['total'] else 0.0
            report += f"- **{ptype.title()}**: {success_rate:.1f}% ({stats['accurate']}/{stats['total']})\n"

        report += "\n## Key Findings\n\n"

        # Identify patterns
        accurate_results = [
            r for r in self.results if r['validation'].get('accuracy', False)]
        inaccurate_results = [
            r for r in self.results if not r['validation'].get('accuracy', False)]

        if accurate_results:
            report += "### Successful Patterns\n"
            for result in accurate_results[:3]:  # Top 3 examples
                report += f"- {result['question']}: Success\n"

        if inaccurate_results:
            report += "\n### Common Errors\n"
            error_types = {}
            for result in inaccurate_results:
                error_type = result['validation'].get('error_type', 'unknown')
                error_types[error_type] = error_types.get(error_type, 0) + 1

            for error, count in error_types.items():
                report += f"- {error.replace('_', ' ').title()}: {count} occurrences\n"

        return report

    def export_results(self, filename: str):
        """Export results to JSON file"""
        ensure_dirs()
        with open(filename, 'w', encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results exported to {filename}")


# ----------------------------- Syracuse Dataset & Validator ----------------------------- #

def wilson_ci(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if trials <= 0:
        return (0.0, 0.0)
    phat = successes / trials
    denom = 1 + (z**2) / trials
    center = (phat + (z**2) / (2 * trials)) / denom
    margin = (z * sqrt((phat * (1 - phat) + (z**2) / (4 * trials)) / trials)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def create_syracuse_2024_dataset():
    """Create Syracuse Women's Lacrosse 2024 dataset from official statistics."""
    syracuse_players = {
        'Player': [
            'Meaghan Tyrrell', 'Olivia Adamson', 'Emma Ward', 'Sam Swart',
            'Payton Rowley', 'Maddy Baxter', 'Savannah Sweitzer', 'Emma Madnick',
            'Jody Cerullo', 'Grace Britton', 'Kendall Rose', 'Kaci Benoit',
            'Sloane Clark', 'Katie Goodale', 'Mackenzie Rich', 'Victoria Reid',
            'Ryann Banks', 'Hallie Simpkins', 'McKenzie Oleen', 'Ruby Hnatkowiak',
            'Sydney Pirreca', 'Carlie Desimone', 'Ally Quirk', 'Tate Paulson',
            'Ryan Johnson', 'Georgia Sexton-Stone', 'Gwenna Gento', 'Ezra Lahan',
            'Ella Bree', 'Talia Waders', 'Jenna Marino', 'Ana Horvit',
            'Delaney Swartout', 'Daniella Guyette'
        ],
        'Jersey': [
            22, 22, 23, 2, 19, 22, 21, 22,
            17, 19, 7, 22, 9, 31, 10, 7,
            4, 22, 21, 22, 6, 9, 5, 1,
            6, 7, 7, 7, 6, 5, 3, 7,
            22, 7
        ],
        'Goals': [
            70, 58, 44, 29, 23, 30, 24, 14,
            11, 6, 8, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0
        ],
        'Assists': [
            32, 25, 37, 18, 15, 6, 9, 13,
            3, 4, 1, 0, 0, 1, 1, 0,
            1, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0
        ],
        'Points': [
            102, 83, 81, 47, 38, 36, 33, 27,
            14, 10, 9, 1, 1, 1, 1, 0,
            1, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0
        ],
        'Shots': [
            115, 109, 90, 53, 55, 64, 54, 42,
            29, 17, 11, 3, 1, 2, 1, 2,
            1, 0, 4, 2, 1, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0
        ],
        'Games_Played': [
            21, 12, 10, 19, 15, 14, 8, 15,
            10, 3, 3, 21, 1, 43, 19, 19,
            0, 26, 3, 1, 10, 0, 0, 1,
            0, 0, 0, 1, 1, 0, 0, 0,
            4, 0
        ]
    }

    df = pd.DataFrame(syracuse_players)

    # Derived player stats
    df['Shooting_Pct'] = np.where(
        df['Shots'] > 0, (df['Goals'] / df['Shots']) * 100, 0.0)
    df['Goals_Per_Game'] = np.where(
        df['Games_Played'] > 0, df['Goals'] / df['Games_Played'], 0.0)
    df['Points_Per_Game'] = np.where(
        df['Games_Played'] > 0, df['Points'] / df['Games_Played'], 0.0)

    # Team stats derived from data (avoid mismatches)
    total_goals = int(df['Goals'].sum())
    total_assists = int(df['Assists'].sum())

    team_stats = {
        'season_record': '16-6',
        'total_games': 22,
        'home_record': '9-2',
        'away_record': '5-2',
        'neutral_record': '2-2',
        'conference_record': '9-1',
        'non_conference_record': '7-5',
        'total_team_goals': total_goals,
        'total_team_assists': total_assists,
        'team_shots': int(df['Shots'].sum()),
        'team_shot_pct': round((total_goals / max(1, int(df['Shots'].sum()))) * 100, 2) if int(df['Shots'].sum()) > 0 else 0,
        'goals_per_game': None,
        'goals_against_per_game': None
    }

    return df, team_stats


class SyracuseDataValidator:
    """Validates LLM responses against real Syracuse Women's Lacrosse 2024 statistics"""

    def __init__(self):
        ensure_dirs()
        self.df, self.team_stats = create_syracuse_2024_dataset()
        self.ground_truth = self._calculate_ground_truth()

        # Save derived dataset snapshot (reproducibility)
        self.df.to_csv(os.path.join(
            "outputs", 'syracuse_lacrosse_2024_real.csv'), index=False)
        with open(os.path.join("data", 'syracuse_data_context.txt'), 'w', encoding="utf-8") as f:
            f.write(self.get_testing_context())

    def _calculate_ground_truth(self) -> Dict[str, Any]:
        """Calculate known statistics from Syracuse data for validation"""
        stats = {}

        # Team-level
        stats['total_games'] = self.team_stats['total_games']
        stats['season_record'] = self.team_stats['season_record']
        stats['wins'] = 16
        stats['losses'] = 6

        # Top performers
        stats['top_scorer'] = self.df.loc[self.df['Goals'].idxmax(), 'Player']
        stats['top_scorer_goals'] = int(self.df['Goals'].max())

        stats['top_assist'] = self.df.loc[self.df['Assists'].idxmax(),
                                          'Player']
        stats['top_assist_count'] = int(self.df['Assists'].max())

        stats['top_points'] = self.df.loc[self.df['Points'].idxmax(), 'Player']
        stats['top_points_count'] = int(self.df['Points'].max())

        # Team totals
        stats['total_goals'] = int(self.df['Goals'].sum())
        stats['total_assists'] = int(self.df['Assists'].sum())
        stats['total_points'] = int(self.df['Points'].sum())

        # Best shooter among qualified (min 10 shots)
        qualified = self.df[self.df['Shots'] >= 10].copy()
        if not qualified.empty:
            best_idx = qualified['Shooting_Pct'].idxmax()
            stats['best_shooter'] = str(qualified.loc[best_idx, 'Player'])
            stats['best_shooting_pct'] = float(
                qualified.loc[best_idx, 'Shooting_Pct'])

        # Active scorers (5+ goals)
        stats['active_scorers'] = int((self.df['Goals'] >= 5).sum())

        # Top-3 goal scorers and computed shooting%
        top3 = self.df.sort_values('Goals', ascending=False).head(3).copy()
        top3['Shooting_Pct_calc'] = np.where(
            top3['Shots'] > 0, (top3['Goals'] / top3['Shots']) * 100.0, 0.0
        )
        stats['top3_shooting'] = [
            {'player': str(row['Player']), 'shooting_pct': round(
                float(row['Shooting_Pct_calc']), 1)}
            for _, row in top3.iterrows()
        ]

        # Wilson 95% CIs for those top-3 shooting percentages
        top3_ci = []
        for _, row in top3.iterrows():
            g, s = int(row['Goals']), int(row['Shots'])
            lo, hi = wilson_ci(g, s)  # in 0..1
            top3_ci.append({'player': str(row['Player']),
                            'shooting_pct_ci': [round(lo * 100, 1), round(hi * 100, 1)]})
        stats['top3_shooting_ci'] = top3_ci

        # Count players with ≥10 goals
        stats['count_ge_10_goals'] = int((self.df['Goals'] >= 10).sum())

        return stats

    def get_testing_context(self) -> str:
        """Formatted data context for LLM testing (pasteable)."""
        context = f"""
Syracuse Women's Lacrosse 2024 Season Statistics:

TEAM RECORD: {self.team_stats['season_record']} ({self.team_stats['total_games']} games)
- Home: {self.team_stats['home_record']}
- Away: {self.team_stats['away_record']}  
- Conference: {self.team_stats['conference_record']}

TOP PERFORMERS:
"""
        top_scorers = self.df.nlargest(
            10, 'Goals')[['Player', 'Goals', 'Assists', 'Points', 'Shots', 'Games_Played']]
        for _, player in top_scorers.iterrows():
            if player['Goals'] > 0:
                shooting_pct = (
                    player['Goals'] / player['Shots'] * 100.0) if player['Shots'] > 0 else 0.0
                context += f"- {player['Player']}: {int(player['Goals'])}G, {int(player['Assists'])}A, {int(player['Points'])}Pts, {int(player['Shots'])} shots ({shooting_pct:.1f}%)\n"

        team_goals = int(self.df['Goals'].sum())
        team_assists = int(self.df['Assists'].sum())
        context += f"\nTEAM TOTALS: {team_goals} Goals, {team_assists} Assists\n"
        return context.strip()

    # ---------- NEW HELPERS FOR TASK 07 ----------

    def usage_fairness_check(self):
        """
        Proxy fairness check by usage (shots): high-usage vs low-usage groups.
        Returns (group rows, disparity).
        """
        df = self.df.copy()
        cutoff = df['Shots'].quantile(0.7)
        df['usage_group'] = np.where(
            df['Shots'] >= cutoff, 'high_usage', 'low_usage')
        grp = df.groupby('usage_group').agg(
            goals=('Goals', 'sum'),
            shots=('Shots', 'sum'),
            players=('Player', 'count')
        ).reset_index()
        grp['shot_pct'] = np.where(
            grp['shots'] > 0, 100 * grp['goals'] / grp['shots'], 0.0)
        high = float(grp.loc[grp['usage_group'] ==
                     'high_usage', 'shot_pct'].values[0])
        low = float(grp.loc[grp['usage_group'] ==
                    'low_usage', 'shot_pct'].values[0])
        disparity = round(high - low, 2)
        return grp.to_dict(orient='records'), disparity

    def robustness_remove_top_n(self, n=1):
        """
        Remove top-n goal scorers and recompute team-level goals and shot%.
        """
        df = self.df.copy().sort_values('Goals', ascending=False)
        removed = df.head(n)['Player'].tolist()
        df2 = df.iloc[n:].copy()
        goals = int(df2['Goals'].sum())
        shots = int(df2['Shots'].sum())
        pct = round(100 * goals / max(1, shots), 2) if shots > 0 else 0.0
        return {'removed_players': removed, 'team_goals': goals, 'team_shot_pct': pct}

    def sensitivity_threshold_players(self, threshold=10):
        """
        Count players with ≥ threshold goals (sensitivity to threshold changes).
        """
        count = int((self.df['Goals'] >= threshold).sum())
        names = self.df.loc[self.df['Goals'] >= threshold, 'Player'].tolist()
        return {'threshold': threshold, 'players_meeting_threshold': count, 'names': names}

    def sanity_checks(self):
        """
        Basic data sanity checks: impossible combos, missingness.
        """
        df = self.df.copy()
        issues = []

        z = df[(df['Goals'] > 0) & (df['Shots'] == 0)]
        if not z.empty:
            issues.append({'type': 'goals_without_shots',
                          'players': z['Player'].tolist()})

        bad = df[df['Shots'] < df['Goals']]
        if not bad.empty:
            issues.append({'type': 'shots_less_than_goals',
                          'players': bad['Player'].tolist()})

        return issues

    def strict_threshold_prompt(self) -> str:
        """
        A stricter prompt to fix the known ≥10 goals list/count pitfall.
        """
        return (
            self.get_testing_context()
            + "\n\nInstruction: Using ONLY the table above, list EXACTLY all players with ≥ 10 goals, by NAME. "
              "Then return the FINAL COUNT and the NAMES. Do not infer or assume. "
              "Output format:\nCount: <number>\nNames: <comma-separated list>"
        )

    # ---------- EXISTING VALIDATION LOGIC (EXTENDED) ----------

    def _extract_numbers_and_percents(self, text: str) -> Tuple[List[float], List[float]]:
        """Extract numeric values (ints/decimals) and percentages from text."""
        nums = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        pcts = re.findall(r'\b\d+(?:\.\d+)?\s*%', text)
        pct_vals = [float(p.rstrip('%').strip()) for p in pcts]
        vals = [float(n) for n in nums]
        return vals, pct_vals

    def _score_strategic_response(self, text: str) -> Dict[str, int]:
        """Score a free-form strategic response using a simple rubric."""
        text_l = text.lower()
        names = [str(n).lower()
                 for n in self.df['Player'].tolist() if str(n).strip()]
        mentions = sum(1 for n in names if n in text_l)
        contains_numbers = bool(re.search(r'\d', text_l))
        specificity = 1 + min(4, mentions // 2) + \
            (1 if contains_numbers else 0)
        specificity = min(5, specificity)

        action_terms = [
            'focus', 'improve', 'increase', 'reduce', 'practice', 'drill', 'scheme',
            'set play', 'assign', 'rotate', 'substitute', 'optimize', 'work on',
            'emphasize', 'target', 'adjust', 'press', 'zone', 'man-to-man', 'transition'
        ]
        actionability = 1 + sum(1 for t in action_terms if t in text_l)
        actionability = max(1, min(5, actionability))

        plausible = True
        top_scorer = str(self.ground_truth['top_scorer']).lower()
        if ('top scorer' in text_l or 'leading scorer' in text_l) and (top_scorer not in text_l):
            plausible = False
        plausibility = 5 if plausible else 2

        return {'specificity': specificity, 'actionability': actionability, 'plausibility': plausibility}

    def _extract_numbers_legacy(self, text: str) -> List[int]:
        """Extract integer values from text (legacy helper kept for basic checks)."""
        numbers = re.findall(r'\b\d+\b', text)
        return [int(n) for n in numbers]

    def validate_response(self, llm_response: str, question_type: str) -> Dict[str, Any]:
        """Validate LLM response against Syracuse ground truth"""
        result = {
            'question_type': question_type,
            'timestamp': now_iso(),
            'accuracy': False,
            'error_type': None,
            'notes': [],
            'expected_answer': None,
            'llm_answer': llm_response[:100] + "..." if len(llm_response) > 100 else llm_response
        }

        legacy_ints = self._extract_numbers_legacy(llm_response)
        response_lower = llm_response.lower()

        if question_type == 'season_record':
            expected = self.ground_truth['season_record']
            result['expected_answer'] = expected
            if expected in llm_response or f"{self.ground_truth['wins']}-{self.ground_truth['losses']}" in llm_response:
                result['accuracy'] = True
            else:
                result['error_type'] = 'incorrect_record'
                result['notes'].append(f"Expected {expected}")

        elif question_type == 'total_games':
            expected = self.ground_truth['total_games']
            result['expected_answer'] = expected
            if legacy_ints and expected in legacy_ints:
                result['accuracy'] = True
            else:
                result['error_type'] = 'incorrect_calculation'
                result['notes'].append(
                    f"Expected {expected}, found numbers: {legacy_ints}")

        elif question_type == 'top_scorer':
            expected_player = self.ground_truth['top_scorer']
            expected_goals = self.ground_truth['top_scorer_goals']
            result['expected_answer'] = f"{expected_player} ({expected_goals} goals)"
            if expected_player.lower() in response_lower:
                result['accuracy'] = True
                if expected_goals in legacy_ints:
                    result['notes'].append("Correctly included goal count")
            else:
                result['error_type'] = 'incorrect_player'
                result['notes'].append(f"Expected {expected_player}")

        elif question_type == 'team_goals':
            expected = self.ground_truth['total_goals']
            result['expected_answer'] = expected
            if legacy_ints and expected in legacy_ints:
                result['accuracy'] = True
            else:
                result['error_type'] = 'incorrect_calculation'
                result['notes'].append(
                    f"Expected {expected}, found: {legacy_ints}")

        elif question_type == 'top_assists':
            expected_player = self.ground_truth['top_assist']
            expected_count = self.ground_truth['top_assist_count']
            result['expected_answer'] = f"{expected_player} ({expected_count} assists)"
            if expected_player.lower() in response_lower:
                result['accuracy'] = True
            else:
                result['error_type'] = 'incorrect_player'
                result['notes'].append(f"Expected {expected_player}")

        elif question_type == 'shooting_analysis':
            expected = [(d['player'].lower(), d['shooting_pct'])
                        for d in self.ground_truth['top3_shooting']]
            result['expected_answer'] = self.ground_truth['top3_shooting']
            vals, pcts = self._extract_numbers_and_percents(llm_response)
            candidates = pcts + vals
            tol = 0.5
            hits = 0
            for name, pct in expected:
                if name in response_lower:
                    if any(abs(float(x) - float(pct)) <= tol for x in candidates):
                        hits += 1
            result['accuracy'] = (hits >= 2)
            if not result['accuracy']:
                result['error_type'] = 'incorrect_shooting_analysis'
                result['notes'].append(
                    f"Matched {hits}/3 expected player% entries (±{tol})")

        elif question_type == 'offensive_balance':
            count_ge_10 = self.ground_truth['count_ge_10_goals']
            result['expected_answer'] = count_ge_10
            vals, _ = self._extract_numbers_and_percents(llm_response)
            ok = any(int(round(v)) == int(count_ge_10) for v in vals)
            result['accuracy'] = ok
            if not ok:
                result['error_type'] = 'incorrect_offensive_depth'
                result['notes'].append(
                    f"Expected {count_ge_10}, found: {vals}")

        elif question_type == 'strategic_analysis':
            scores = self._score_strategic_response(llm_response)
            result['expected_answer'] = 'Rubric-based (Specificity, Actionability, Plausibility >= 3)'
            result['notes'].append(f"Scores: {scores}")
            result['accuracy'] = (scores['specificity'] >= 3 and
                                  scores['actionability'] >= 3 and
                                  scores['plausibility'] >= 3)
            if not result['accuracy']:
                result['error_type'] = 'insufficient_rubric_scores'

        return result

    def print_ground_truth(self):
        """Print the correct answers for validation"""
        print("=== SYRACUSE 2024 VALIDATION ANSWERS ===")
        print(f"Season Record: {self.ground_truth['season_record']}")
        print(f"Total Games: {self.ground_truth['total_games']}")
        print(
            f"Top Scorer: {self.ground_truth['top_scorer']} ({self.ground_truth['top_scorer_goals']} goals)")
        print(
            f"Top Assists: {self.ground_truth['top_assist']} ({self.ground_truth['top_assist_count']} assists)")
        print(f"Total Team Goals: {self.ground_truth['total_goals']}")
        print(f"Total Team Assists: {self.ground_truth['total_assists']}")
        print(f"Best Shooter (≥10 shots): {self.ground_truth.get('best_shooter', 'N/A')} "
              f"({self.ground_truth.get('best_shooting_pct', 0):.1f}%)")
        print(
            f"Active Scorers (5+ goals): {self.ground_truth['active_scorers']}")
        print("Top-3 Shooting %:", self.ground_truth['top3_shooting'])
        print("Top-3 Shooting % CIs:", self.ground_truth['top3_shooting_ci'])
        print("Players with ≥10 goals:",
              self.ground_truth['count_ge_10_goals'])


# ----------------------------- Prompt Generation ----------------------------- #

def generate_syracuse_test_prompts(validator: SyracuseDataValidator) -> List[Dict[str, str]]:
    """Generate test prompts using Syracuse data"""
    context = validator.get_testing_context()

    prompts = [
        # BASIC
        {'type': 'basic', 'question_type': 'season_record',
         'prompt': f"{context}\n\nQuestion: What was Syracuse Women's Lacrosse team record for the 2024 season?"},
        {'type': 'basic', 'question_type': 'total_games',
         'prompt': f"{context}\n\nQuestion: How many total games did Syracuse play in the 2024 season?"},
        {'type': 'basic', 'question_type': 'top_scorer',
         'prompt': f"{context}\n\nQuestion: Who was Syracuse's leading goal scorer in 2024 and how many goals did they score?"},
        {'type': 'basic', 'question_type': 'team_goals',
         'prompt': f"{context}\n\nQuestion: How many total goals did the Syracuse team score in 2024?"},
        {'type': 'basic', 'question_type': 'top_assists',
         'prompt': f"{context}\n\nQuestion: Who led Syracuse in assists in 2024?"},

        # INTERMEDIATE
        {'type': 'intermediate', 'question_type': 'shooting_analysis',
         'prompt': f"{context}\n\nQuestion: Calculate the shooting percentage for Syracuse's top 3 goal scorers. Who was most efficient?"},
        {'type': 'intermediate', 'question_type': 'offensive_balance',
         'prompt': f"{context}\n\nQuestion: Analyze Syracuse's offensive balance. How many players scored at least 10 goals? What does this suggest about their offensive depth?"},

        # COMPLEX
        {'type': 'complex', 'question_type': 'strategic_analysis',
         'prompt': (f"{context}\n\nAs a coach analyzing Syracuse's 2024 season (16-6 record), answer:\n"
                    "1. What were the team's main offensive strengths?\n"
                    "2. If you wanted to improve to 18-4 next season, what specific areas would you focus on?\n"
                    "3. Which player had the biggest impact beyond just goals scored?")}
    ]
    return prompts


# ----------------------------- Interactive Helpers ----------------------------- #

def show_test_prompt(prompts):
    print("\nAvailable prompts:")
    for i, prompt in enumerate(prompts):
        print(f"{i+1}. {prompt['type'].title()}: {prompt['question_type']}")
    try:
        choice = int(input("Select prompt number: ")) - 1
        if 0 <= choice < len(prompts):
            print(f"\n{'='*60}")
            print(f"PROMPT TYPE: {prompts[choice]['type'].upper()}")
            print(f"QUESTION: {prompts[choice]['question_type']}")
            print(f"{'='*60}")
            print(prompts[choice]['prompt'])
            print(f"{'='*60}")
        else:
            print("Invalid selection")
    except ValueError:
        print("Please enter a valid number")


def show_all_prompts(prompts):
    for i, prompt in enumerate(prompts):
        print(f"\n{'='*40}")
        print(
            f"PROMPT {i+1}: {prompt['type'].upper()} - {prompt['question_type']}")
        print(f"{'='*40}")
        print(prompt['prompt'][:200] +
              "..." if len(prompt['prompt']) > 200 else prompt['prompt'])


def validate_llm_response(validator: SyracuseDataValidator, analyzer: ResultsAnalyzer):
    print("\nQuestion types available:")
    print("- season_record")
    print("- total_games")
    print("- top_scorer")
    print("- team_goals")
    print("- top_assists")
    print("- shooting_analysis")
    print("- offensive_balance")
    print("- strategic_analysis")

    question_type = input("Enter question type: ").strip()
    llm_response = input("Paste LLM response here: ").strip()

    result = validator.validate_response(llm_response, question_type)

    print(f"\n{'='*50}")
    print("VALIDATION RESULT:")
    print(f"Accuracy: {'✓ CORRECT' if result['accuracy'] else '✗ INCORRECT'}")
    print(f"Expected: {result['expected_answer']}")
    if result['error_type']:
        print(f"Error Type: {result['error_type']}")
    if result['notes']:
        print(f"Notes: {'; '.join(str(n) for n in result['notes'])}")
    print(f"{'='*50}")

    analyzer.add_result(question_type, question_type, llm_response, result)


# ----------------------------- Task 07 Exporters ----------------------------- #

def export_task07_artifacts(validator: SyracuseDataValidator, analyzer: ResultsAnalyzer):
    """
    One-click exporter:
    - ground truth
    - fairness proxy
    - robustness & sensitivity
    - sanity checks
    - summary report
    - strict threshold prompt
    """
    ensure_dirs()

    # Ground truth snapshot
    save_json(os.path.join("outputs", "ground_truth.json"),
              validator.ground_truth)

    # Fairness proxy
    groups, disparity = validator.usage_fairness_check()
    save_json(os.path.join("outputs", "fairness_usage.json"),
              {"groups": groups, "shot_pct_disparity_high_minus_low": disparity})

    # Robustness (remove top 1 & 2)
    r1 = validator.robustness_remove_top_n(1)
    r2 = validator.robustness_remove_top_n(2)
    save_json(os.path.join("outputs", "robustness.json"),
              {"remove_top1": r1, "remove_top2": r2})

    # Sensitivity (≥10 vs ≥12)
    s10 = validator.sensitivity_threshold_players(10)
    s12 = validator.sensitivity_threshold_players(12)
    save_json(os.path.join("outputs", "sensitivity.json"),
              {"threshold_10": s10, "threshold_12": s12})

    # Sanity checks
    sanity = validator.sanity_checks()
    save_json(os.path.join("outputs", "sanity_checks.json"), sanity)

    # Analyzer results and summary
    analyzer.export_results(os.path.join("outputs", "validation_results.json"))
    summary = analyzer.generate_summary_report()
    with open(os.path.join("outputs", "syracuse_testing_report.md"), "w", encoding="utf-8") as f:
        f.write(summary)

    # Strict threshold prompt file
    strict_prompt = validator.strict_threshold_prompt()
    with open(os.path.join("prompts", "threshold_prompts_strict.txt"), "w", encoding="utf-8") as f:
        f.write(strict_prompt)

    print("\n✅ Task 07 artifacts exported to /outputs and strict prompt saved to /prompts.")


# ----------------------------- Main Runner ----------------------------- #

def run_syracuse_testing():
    """Interactive testing session with Syracuse data"""
    print("=== SYRACUSE WOMEN'S LACROSSE 2024 LLM TESTING (Task 07) ===")
    validator = SyracuseDataValidator()
    analyzer = ResultsAnalyzer()

    validator.print_ground_truth()

    test_prompts = generate_syracuse_test_prompts(validator)
    print(f"\n=== {len(test_prompts)} TEST PROMPTS GENERATED ===")

    while True:
        print("\n" + "="*60)
        print("SYRACUSE TESTING MENU:")
        print("1. Show test prompt")
        print("2. Validate LLM response")
        print("3. Show all prompts")
        print("4. Generate summary report (preview)")
        print("5. Export ALL Task 07 artifacts")
        print("6. Show STRICT threshold prompt (≥10 goals)")
        print("7. Exit")

        choice = input("\nEnter choice (1-7): ").strip()

        if choice == '1':
            show_test_prompt(test_prompts)
        elif choice == '2':
            validate_llm_response(validator, analyzer)
        elif choice == '3':
            show_all_prompts(test_prompts)
        elif choice == '4':
            report = analyzer.generate_summary_report()
            print("\n" + report)
        elif choice == '5':
            export_task07_artifacts(validator, analyzer)
        elif choice == '6':
            print("\n" + "="*60)
            print("STRICT THRESHOLD PROMPT")
            print("="*60)
            print(validator.strict_threshold_prompt())
            print("="*60)
        elif choice == '7':
            break
        else:
            print("Invalid selection.")


def main():
    ensure_dirs()
    print("Syracuse Women's Lacrosse 2024 LLM Testing Framework (Task 07)")

    while True:
        print("""
LAUNCHER MENU
1) Interactive tester (shows 1–7 menu)
2) Export ALL Task 07 artifacts (non-interactive quick run)
3) Show STRICT threshold prompt (≥10 goals)
4) Print ground truth answers
5) Quit
""")
        choice = input("Enter choice [1-5] (default: 1): ").strip() or "1"

        if choice == "1":
            run_syracuse_testing()
        elif choice == "2":
            validator = SyracuseDataValidator()
            analyzer = ResultsAnalyzer()
            export_task07_artifacts(validator, analyzer)
        elif choice == "3":
            validator = SyracuseDataValidator()
            print("\n" + "="*60)
            print("STRICT THRESHOLD PROMPT")
            print("="*60)
            print(validator.strict_threshold_prompt())
            print("="*60 + "\n")
        elif choice == "4":
            validator = SyracuseDataValidator()
            validator.print_ground_truth()
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid selection. Please choose 1–5.")


if __name__ == "__main__":
    main()
