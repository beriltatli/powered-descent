import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import report  # noqa: E402


def test_report_matches_results():
    # REPORT.md is rendered from results/*.json; any hand edit or stale number shows up as a diff.
    assert (ROOT / "REPORT.md").read_text() == report.render()


def test_template_has_no_literal_numbers_outside_placeholders():
    # Numbers in prose must come from results/, so the template may only carry numbers that are part of
    # the problem statement (equation numbers, the brief's thresholds, paper values). Those are listed
    # here explicitly; anything else is a number someone typed in by hand.
    allowed = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17",
               "18", "19", "20", "21", "22", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35",
               "36", "37", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53",
               "54", "55", "56", "57", "58", "59", "60", "61", "62", "86", "90", "99", "95", "100", "500", "1000",
               "2007", "2013", "2014", "1500", "2000", "75", "0", "400", "1905", "1505", "225", "3.1", "0.3", "0.8",
               "3.7114", "69", "293.6", "78", "303", "72", "387.9", "390.4", "81", "399.5", "1e-4", "1e-3",
               "1e-10", "1e-8", "0.05", "0.95", "5.26", "10.5", "0.25", "1.0", "1e-5", "1e-6",
               "1e-7", "1356", "1357", "1361", "1363", "1364", "1365", "1359", "1360", "1362", "1355",
               "1358", "24", "23", "38", "39", "8",
               # design constants and thresholds from config.yaml or the brief, not results
               "0.97", "0.99", "150", "1e-9", "80", "120", "200",
               # Malyuta et al. tutorial: equation number and publication years of the cited sources
               "106", "2011", "2022"}
    tpl = (ROOT / "scripts" / "report_template.md").read_text()
    prose = re.sub(r"\$\{?\w+\}?", "", tpl)
    nums = set(re.findall(r"(?<![\w.$])\d+(?:\.\d+)?(?:e-?\d+)?(?![\w])", prose))
    assert nums <= allowed, sorted(nums - allowed)
