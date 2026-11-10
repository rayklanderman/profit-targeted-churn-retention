"""Build the 15-slide deck (PPTX) from outputs/results.json and outputs/figs/*.png.
Run after analysis.py.  Usage: python build_deck.py [AccountName]
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
from lxml import etree

ACCOUNT = sys.argv[1] if len(sys.argv) > 1 else "RealDevRay"
OUT = Path("outputs"); FIG = OUT / "figs"
R = json.loads((OUT / "results.json").read_text())

NAVY, RED, TEAL, GREY, GOLD = "1F3A5F", "D1495B", "2A9D8F", "8D99AE", "EDAE49"
rgb = lambda h: RGBColor.from_string(h)
INK = "222B38"; LIGHT = "F3F5F9"

# ---------------------------------------------------------------- market chart (external data, cited on slide)
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 15})
yrs = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
conn = [400.2, 421.8, 442.5, 468.9, 498.9, 523.0, 558.1, 579.0, 599.0]
fig, ax = plt.subplots(figsize=(6.4, 4.2))
bars = ax.bar([str(y) for y in yrs], conn, color=["#" + NAVY] * 8 + ["#" + RED])
for b, v in zip(bars, conn):
    ax.text(b.get_x() + b.get_width() / 2, v + 6, f"{v:.0f}", ha="center", fontsize=10.5)
ax.set_ylabel("US wireless connections (millions)"); ax.set_ylim(0, 690); ax.set_title("Connections keep growing, people do not")
ax.tick_params(axis="x", labelsize=10.5)
plt.tight_layout(); plt.savefig(FIG / "fig_market.png", dpi=200); plt.close()

# ---------------------------------------------------------------- helpers
prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
page = {"n": 0}


def new_slide(title, sources=None):
    s = prs.slides.add_slide(BLANK); page["n"] += 1
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.28), Inches(12.3), Inches(0.95)); tf = tb.text_frame; tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; r = p.add_run(); r.text = title; r.font.size = Pt(27); r.font.bold = True; r.font.color.rgb = rgb(NAVY); r.font.name = "Calibri"
    ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.25), Inches(1.2), Inches(0.06)); ln.fill.solid(); ln.fill.fore_color.rgb = rgb(RED); ln.line.fill.background()
    ft = s.shapes.add_textbox(Inches(0.5), Inches(7.05), Inches(11.2), Inches(0.35)); fp = ft.text_frame.paragraphs[0]
    r = fp.add_run(); r.text = "" if sources == "" else (("Sources: " + sources) if sources else "Source: own analysis of Company A data"); r.font.size = Pt(10); r.font.color.rgb = rgb(GREY); r.font.name = "Calibri"
    nb = s.shapes.add_textbox(Inches(12.2), Inches(7.05), Inches(0.7), Inches(0.35)); np_ = nb.text_frame.paragraphs[0]; np_.alignment = PP_ALIGN.RIGHT
    r = np_.add_run(); r.text = str(page["n"]); r.font.size = Pt(10); r.font.color.rgb = rgb(GREY)
    return s


def add_text(s, x, y, w, h, items, size=16, color=INK, bullet=False, anchor=MSO_ANCHOR.TOP, align=PP_ALIGN.LEFT, spacing=6):
    """items: list of str or (str, dict(size,bold,color,bullet))"""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05); tf.margin_top = tf.margin_bottom = Inches(0.03)
    for i, it in enumerate(items):
        txt, o = (it, {}) if isinstance(it, str) else it
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = o.get("align", align); p.space_after = Pt(o.get("space", spacing))
        parts = txt.split("**")
        for k, part in enumerate(parts):
            if not part: continue
            r = p.add_run(); r.text = part; r.font.name = "Calibri"; r.font.size = Pt(o.get("size", size))
            r.font.bold = bool(o.get("bold", False)) or (k % 2 == 1); r.font.color.rgb = rgb(o.get("color", color))
        if o.get("bullet", bullet):
            pPr = p._p.get_or_add_pPr(); pPr.set("marL", str(int(Inches(0.25)))); pPr.set("indent", str(-int(Inches(0.25))))
            for tag in ("a:buNone", "a:buChar"):
                for e in pPr.findall(qn(tag)): pPr.remove(e)
            bu = etree.SubElement(pPr, qn("a:buChar")); bu.set("char", "•")
    return tb


def add_img(s, name, x, y, w, h):
    path = FIG / name; im = Image.open(path); ar = im.width / im.height
    if w / h > ar: ww, hh = h * ar, h
    else: ww, hh = w, w / ar
    return s.shapes.add_picture(str(path), Inches(x + (w - ww) / 2), Inches(y + (h - hh) / 2), Inches(ww), Inches(hh))


def card(s, x, y, w, h, big, small, color=NAVY, big_size=34, small_size=13):
    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)); box.adjustments[0] = 0.08
    box.fill.solid(); box.fill.fore_color.rgb = rgb(LIGHT); box.line.fill.background()
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y + 0.12), Inches(0.08), Inches(h - 0.24)); bar.fill.solid(); bar.fill.fore_color.rgb = rgb(color); bar.line.fill.background()
    add_text(s, x + 0.2, y + 0.05, w - 0.3, h - 0.1, [(big, {"size": big_size, "bold": True, "color": color, "space": 2}), (small, {"size": small_size, "color": INK})], anchor=MSO_ANCHOR.MIDDLE)


def box(s, x, y, w, h, text, fill=NAVY, color="FFFFFF", size=14, bold=True):
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)); b.adjustments[0] = 0.12
    b.fill.solid(); b.fill.fore_color.rgb = rgb(fill); b.line.fill.background()
    tf = b.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.margin_left = tf.margin_right = Inches(0.08)
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = line; r.font.size = Pt(size if i == 0 else size - 2); r.font.bold = bold if i == 0 else False; r.font.color.rgb = rgb(color); r.font.name = "Calibri"
    return b


def arrow(s, x, y, w=0.4, h=0.3, color=GREY):
    a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h)); a.fill.solid(); a.fill.fore_color.rgb = rgb(color); a.line.fill.background()


def table(s, x, y, w, h, rows, col_w=None, size=12, header_fill=NAVY, bold_last=False, hl_row=None):
    nr, nc = len(rows), len(rows[0]); gt = s.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(h)); t = gt.table
    if col_w:
        for j, cw in enumerate(col_w): t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j); c.text = ""; p = c.text_frame.paragraphs[0]; r = p.add_run(); r.text = str(val); r.font.size = Pt(size); r.font.name = "Calibri"
            c.margin_left = c.margin_right = Inches(0.06); c.margin_top = c.margin_bottom = Inches(0.03); c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            if i == 0:
                c.fill.fore_color.rgb = rgb(header_fill); r.font.color.rgb = rgb("FFFFFF"); r.font.bold = True
            else:
                c.fill.fore_color.rgb = rgb("FDECEF" if hl_row == i else ("FFFFFF" if i % 2 else LIGHT)); r.font.color.rgb = rgb(INK)
                r.font.bold = (hl_row == i) or (bold_last and i == nr - 1)
            if j > 0: p.alignment = PP_ALIGN.CENTER
    return t


# ---------------------------------------------------------------- numbers
mt = {r["model"]: r for r in R["model_table"]}
FINAL = R["final_model"]; base = R["strategies_base"]; V = base["Value-aware model (contact if expected gain > 0)"]
RK = base["Risk-only model (contact if p > break-even)"]; RU = base["Rule of thumb (tenure 10-12 mo or handset 300-420 d)"]
EV = base["Contact everyone"]; sc = R["scenarios"]
cl = R["cliff_months"]; ce = R["cliff_eqpdays"]
auc = R["final_auc_holdout"]; ci = R["final_auc_ci"]
profit_m = V["profit"]; profit_y = profit_m * 12
k = lambda v: f"${v/1000:,.0f}k"
usd = lambda v: f"${v:,.0f}" if v >= 0 else f"-${-v:,.0f}"
pct = lambda v, d=0: f"{v*100:.{d}f}%"
_s1 = R["sens1"]; _gi = _s1["gammas"].index(0.3)
leak_low = _s1["profit"][_gi][_s1["alphas"].index(0.05)]; leak_high = _s1["profit"][_gi][_s1["alphas"].index(1.0)]

# ================================================================= 1. title
s = prs.slides.add_slide(BLANK); page["n"] += 1
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height); bg.fill.solid(); bg.fill.fore_color.rgb = rgb(NAVY); bg.line.fill.background()
acc = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(3.55), Inches(1.6), Inches(0.08)); acc.fill.solid(); acc.fill.fore_color.rgb = rgb(RED); acc.line.fill.background()
add_text(s, 0.8, 1.3, 11.6, 2.2, [("Catch the exit before it happens", {"size": 46, "bold": True, "color": "FFFFFF", "space": 4}),
                                  ("A profit-targeted churn-retention engine for Company A (US wireless)", {"size": 24, "color": "DDE4F0"})], anchor=MSO_ANCHOR.BOTTOM)
add_text(s, 0.8, 3.85, 11.6, 2.2, [("Proof-of-Concept proposal for the executive team", {"size": 20, "color": "FFFFFF", "bold": True}),
                                   ("GCI World 2026 Winter Final Assignment  |  Matsuo-Iwasawa Lab, The University of Tokyo", {"size": 15, "color": "DDE4F0"}),
                                   ("Prepared by Raymond Klanderman  |  November 2026", {"size": 15, "color": "DDE4F0"})])

# ================================================================= 2. executive summary
s = new_slide("Executive summary: churn is predictable at two lifecycle moments, and targeting by profit makes it pay", "[3][4][5] market churn; own analysis of Company A data; [8] cost-benefit framework")
def panel(x, ttl, body, colr):
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.6), Inches(4.0), Inches(2.1)); b.adjustments[0] = 0.06
    b.fill.solid(); b.fill.fore_color.rgb = rgb(LIGHT); b.line.fill.background()
    br = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(1.72), Inches(0.08), Inches(1.86)); br.fill.solid(); br.fill.fore_color.rgb = rgb(colr); br.line.fill.background()
    add_text(s, x + 0.22, 1.68, 3.65, 1.95, [(ttl, {"size": 20, "bold": True, "color": colr, "space": 4}), (body, {"size": 14})])


panel(0.5, "The problem", "A mature US market (1.7 connections per person) means growth must come from **keeping** customers. Carriers lose about 1% (postpaid) to 3% (prepaid) of customers every month.", RED)
panel(4.67, "What we found", f"Churn jumps from **{pct(cl['before_5_8'][0])} to {pct(cl['peak_10_12'][0])}** at months 10-12 and from **{pct(ce['before_lt300'][0])} to {pct(ce['after_300_420'][0])}** once the handset is 300-420 days old.", NAVY)
panel(8.84, "What we propose", "A scoring engine that picks **who to contact and when**, only where the expected profit is positive, starting with an 8-week randomised pilot.", TEAL)
card(s, 0.5, 3.95, 3.95, 1.45, f"{auc:.3f}", f"ROC-AUC of the {FINAL.split(' (')[0]} model on 20,000 unseen customers (baseline {R['baseline_auc_holdout']:.3f})", NAVY, big_size=36)
card(s, 4.69, 3.95, 3.95, 1.45, usd(profit_m) + " / month", f"net profit per 100,000 customers at base-case assumptions (about {usd(profit_y)} a year)", TEAL, big_size=28)
card(s, 8.88, 3.95, 3.95, 1.45, f"{V['contacts']:,.0f} contacts", f"out of 100,000 customers get an offer; contacting everyone would lose {usd(-EV['profit'])} a month", RED, big_size=28)
add_text(s, 0.5, 5.65, 12.3, 1.3, [("Why it matters", {"size": 16, "bold": True, "color": NAVY}),
                                  (f"The model alone is not the product, **offer design is**. Profit falls from {usd(leak_low)} a month when few loyal customers claim the offer (5%) to {usd(leak_high)} when all of them do, and the pessimistic scenario is only break-even ({usd(sc['Pessimistic']['profit'])}). The pilot therefore measures offer take-up before any scale-up.", {"size": 14.5})], spacing=3)

# ================================================================= 3. market
s = new_slide("Market: a saturated US market makes keeping a customer worth more than winning one", "[1] CTIA 2025; [2] Light Reading 2026; [3][4][5] company reports; [6] HBR 2014; [7] J. Marketing Research")
add_img(s, "fig_market.png", 0.4, 1.5, 6.3, 4.4)
card(s, 7.0, 1.55, 2.85, 1.55, "0.9-1.0%", "monthly postpaid phone churn at T-Mobile, Verizon, AT&T (2025) [3][4][5]", NAVY, big_size=28, small_size=12)
card(s, 10.0, 1.55, 2.85, 1.55, "2.7-3.7%", "monthly prepaid churn at the same carriers, about 3x higher [3][4][5]", RED, big_size=28, small_size=12)
card(s, 7.0, 3.25, 2.85, 1.55, "5-25x", "cost of acquiring vs retaining a customer across industries (HBR) [6]", TEAL, big_size=28, small_size=12)
card(s, 10.0, 3.25, 2.85, 1.55, "~3x", "$160 acquisition vs $53 retention per quarter in 41 countries' wireless data [7]", GOLD, big_size=28, small_size=12)
add_text(s, 7.0, 5.0, 5.9, 1.9, [("So what for Company A", {"size": 15, "bold": True, "color": NAVY}),
                                 ("Connections grew ~5% a year but already exceed the population, so net adds now come from devices and rivals' customers [1][2].", {"size": 13, "bullet": True}),
                                 ("Churn rises as customers roll off device-financing plans and carriers answer with large discounts [12], so timing matters.", {"size": 13, "bullet": True})], spacing=3)
add_text(s, 0.5, 5.95, 6.2, 1.0, ["Note: 2025 total from Light Reading citing CTIA; 46% of 2025 connections are not phones [2]."], size=11, color=GREY)

# ================================================================= 4. client and data
s = new_slide("The client and the data: 100,000 customers, 98 fields, one clear target", "Company A dataset overview and tutorial [16]; [9] on over-sampled churn samples; own analysis")
add_text(s, 0.5, 1.55, 6.2, 5.3, [("Situation", {"size": 17, "bold": True, "color": NAVY}),
                                  ("Company A is a US wireless carrier with a large usage and billing history but **no in-house ML capability** [16].", {"bullet": True}),
                                  ("Two linked tables (Client, Record), one row per customer, joined on Customer_ID (1:1, no duplicates).", {"bullet": True}),
                                  ("Contents: tenure, plan price, usage and call quality, handset history, credit class, region and household demographics.", {"bullet": True}),
                                  ("Why churn as the target", {"size": 17, "bold": True, "color": NAVY}),
                                  ("It is the direct driver of revenue loss and acquisition spend, it is observed for every customer, and the market data above shows retention is the cheapest lever [6][7].", {"bullet": True}),
                                  ("Churn here = left within 31-60 days of the observation date.", {"bullet": True})], size=14, spacing=5)
add_img(s, "fig_missing.png", 6.9, 1.5, 6.0, 3.6)
add_text(s, 6.9, 5.15, 6.0, 1.8, [("Two data cautions", {"size": 15, "bold": True, "color": RED}),
                                  (f"**Over-sampled target:** {pct(R['churn_rate_sample'],1)} of the sample churned, but real carriers lose ~1-3% a month. Classic practice for model building [9]; we re-weight to a realistic rate before computing money.", {"size": 13, "bullet": True}),
                                  ("**Sparse demographics:** 22-49% missing; we keep the signal by counting how many fields are missing.", {"size": 13, "bullet": True})], spacing=3)

# ================================================================= 5. EDA cliffs
s = new_slide(f"EDA 1: churn jumps from {pct(cl['before_5_8'][0])} to {pct(cl['peak_10_12'][0])} at month 10, and again when the handset is ~300 days old", "Own analysis of Company A data (n = 100,000); [11][12] on contract/upgrade timing")
add_img(s, "fig_cliffs.png", 0.4, 1.5, 12.5, 4.1)
add_text(s, 0.5, 5.7, 6.1, 1.3, [("What it means", {"size": 15, "bold": True, "color": NAVY}),
                                 ("Churn is not spread evenly: it clusters in **two predictable windows** in the customer's lifecycle - the first anniversary and the handset-upgrade window.", {"size": 13, "bullet": True})], spacing=3)
add_text(s, 6.8, 5.7, 6.1, 1.3, [("Hypothesis (to be tested in the pilot)", {"size": 15, "bold": True, "color": RED}),
                                 ("Customers re-evaluate their carrier when a contract term or financing period ends, or an upgrade becomes available [11][12]. In a Canadian study, customers who upgraded their handset once were 22.5% less likely to churn [11].", {"size": 13, "bullet": True})], spacing=3)

# ================================================================= 6. EDA heat + signals
s = new_slide("EDA 2: the two windows compound, while service quality and care calls explain little", "Own analysis of Company A data (n = 100,000)")
add_img(s, "fig_heatmap.png", 0.4, 1.45, 5.5, 2.95)
add_img(s, "fig_other_signals.png", 1.6, 4.45, 10.1, 2.55)
sp = R["octile_spread"]
add_text(s, 6.2, 1.5, 6.7, 1.3, [("Reading the lower charts", {"size": 14, "bold": True, "color": NAVY}),
                                 (f"Falling usage and cheaper handsets go with churn (gap between the lowest and highest group: {sp['change_mou']*100:.0f} and {sp['hnd_price']*100:.0f} percentage points).", {"size": 12.5, "bullet": True}),
                                 (f"Care calls and dropped/blocked calls move it less ({sp['custcare_Mean']*100:.0f} and {sp['drop_blk_Mean']*100:.0f} points): network quality is a secondary story here.", {"size": 12.5, "bullet": True})], spacing=3)
add_text(s, 6.2, 2.9, 6.7, 1.5, [("Takeaway", {"size": 14, "bold": True, "color": NAVY}),
                                  (f"Customers 10-12 months in with a 300-420 day-old handset churn at {pct(R['heatmap_rate']['300-420 d']['10-12 mo'])} ({R['heatmap_n']['300-420 d']['10-12 mo']:,.0f} customers). The same tenure band with a newer handset churns much less.", {"size": 12.5, "bullet": True}),
                                  (f"Regions differ too ({R['area_low'][0].title()} {pct(R['area_low'][1])} vs {R['area_high'][0].title()} {pct(R['area_high'][1])}).", {"size": 12.5, "bullet": True}),
                                  ("Business implication: spend on **timing and offers**, not on network fixes, for this problem.", {"size": 12.5, "bold": True, "color": RED})], spacing=3)

# ================================================================= 7. problem definition + approach
s = new_slide("Problem definition: rank customers by churn risk, then by expected profit", "[8] Verbraken et al. 2013 (cost-benefit framework); own design")
steps = [("1  Data", "100k customers\n98 fields"), ("2  Features", "Trends, rates,\nlifecycle timing"), ("3  Model", "Churn probability\nper customer"), ("4  Money", "Value x risk\nminus costs"), ("5  Action", "Contact only if\nexpected gain > 0")]
x0 = 0.55
for i, (a, b) in enumerate(steps):
    box(s, x0 + i * 2.5, 1.75, 2.1, 1.25, a + "\n" + b, fill=[NAVY, NAVY, TEAL, GOLD, RED][i], size=16)
    if i < 4: arrow(s, x0 + i * 2.5 + 2.14, 2.25, 0.32, 0.28)
add_text(s, 0.5, 3.35, 6.1, 3.6, [("Business objective", {"size": 16, "bold": True, "color": NAVY}),
                                  ("Maximise the net profit of a monthly retention campaign. The contact list is chosen by expected gain.", {"bullet": True}),
                                  ("ML task", {"size": 16, "bold": True, "color": NAVY}),
                                  ("Supervised binary classification: P(churn within 31-60 days).", {"bullet": True}),
                                  ("Target and why", {"size": 16, "bold": True, "color": NAVY}),
                                  ("Churn is the direct driver of revenue loss, it is observed for every customer, and keeping a customer is cheaper than replacing one [6][7].", {"bullet": True})], size=14, spacing=4)
add_text(s, 6.8, 3.35, 6.1, 3.6, [("How we will judge success", {"size": 16, "bold": True, "color": NAVY}),
                                  ("**Model metric: ROC-AUC** - how well it ranks churners above stayers; not distorted by the over-sampled 50/50 sample. Supported by top-decile lift and PR-AUC.", {"bullet": True}),
                                  ("**Business metric: net profit per 100,000 customers** after re-weighting to a realistic churn rate (2% base case).", {"bullet": True}),
                                  ("**Fair testing:** all choices are tuned on 80% of the data with 5-fold cross-validation; a 20% hold-out of 20,000 customers is opened once at the end.", {"bullet": True})], size=14, spacing=4)

# ================================================================= 8. features & method
s = new_slide("Method: simple ratios that describe behaviour, and four model families compared", "[13] LightGBM; [14] XGBoost; [15] scikit-learn; own design")
add_text(s, 0.5, 1.55, 6.1, 5.4, [("Feature engineering (98 raw columns to 113 features)", {"size": 16, "bold": True, "color": NAVY}),
                                  ("**Trends:** last 3/6-month usage, calls and revenue vs life-time average (is the customer fading?).", {"bullet": True}),
                                  ("**Quality rates:** failed and dropped calls per attempt; care calls per call.", {"bullet": True}),
                                  ("**Lifecycle:** handset age relative to tenure; handsets per year; flags for the two windows.", {"bullet": True}),
                                  ("**Money:** overage share of bill, revenue per minute, handset price vs plan price.", {"bullet": True}),
                                  ("**Missing data:** a count of empty demographic fields, so the gaps themselves carry signal.", {"bullet": True}),
                                  (f"Result: a small gain (LightGBM AUC {R['ablation_fe']['lgbm_raw_features']:.3f} on raw columns, {R['ablation_fe']['lgbm_engineered_features']:.3f} with engineered features).", {"bullet": True, "color": TEAL, "bold": True})], size=14.5, spacing=6)
rows = [["Model", "Why we tried it"],
        ["Logistic regression", "Transparent linear baseline the client can audit"],
        ["Random forest", "Many decision trees voting; robust default"],
        ["XGBoost", "Boosted trees; learns interactions such as tenure x handset age"],
        ["LightGBM", "Fast boosted trees with native categories"],
        ["Blend", "Average of the three best, to reduce variance"]]
table(s, 6.8, 1.65, 6.1, 2.7, rows, col_w=[1.9, 4.2], size=12.5)
add_text(s, 6.8, 4.55, 6.1, 2.4, [("Training protocol", {"size": 16, "bold": True, "color": NAVY}),
                                  ("80% development / 20% untouched hold-out (stratified, seed 42).", {"bullet": True}),
                                  ("Hyper-parameters (tree size, number of rounds) chosen by 5-fold cross-validation with early stopping on the development set only.", {"bullet": True}),
                                  ("Reference = default XGBoost on raw columns, i.e. the tutorial approach.", {"bullet": True})], size=13.5, spacing=4)

# ================================================================= 9. model results
s = new_slide(f"Results: LightGBM and XGBoost tie at a moderate ROC-AUC of {auc:.3f} on unseen customers, {(auc - R['baseline_auc_holdout']):.3f} above the baseline", "Own analysis; hold-out n = 20,000 customers; [17][18] benchmark context")
add_img(s, "fig_models.png", 0.35, 1.45, 7.4, 3.0)
order = ["Baseline: XGBoost defaults, raw columns", "Logistic regression", "Random forest", "XGBoost (tuned)", "LightGBM (tuned)", "Blend (LR + XGB + LGBM)"]
rows = [["Model", "Hold-out ROC-AUC", "5-fold CV AUC", "Lift top 10%"]]
for n in order:
    m = mt[n]; rows.append([n.replace("Baseline: ", "Baseline - "), f"{m['auc']:.3f}", f"{m['cv_auc']:.3f}" + (f" ± {m['cv_auc_sd']:.3f}" if m.get("cv_auc_sd") else ""), f"{m['lift10']:.2f}x"])
hl = order.index(FINAL) + 1 if FINAL in order else None
table(s, 7.9, 1.55, 5.0, 3.0, rows, col_w=[2.1, 1.0, 1.1, 0.8], size=11, hl_row=hl)
add_text(s, 0.5, 4.65, 6.3, 2.3, [("How to read it", {"size": 15, "bold": True, "color": NAVY}),
                                  (f"**Chosen model: {FINAL}** (best out-of-fold AUC, chosen without looking at the hold-out). Hold-out ROC-AUC **{auc:.3f}** (95% interval {ci[0]:.3f}-{ci[1]:.3f}); top-10% of risk scores contain **{R['final_lift10']:.1f}x** more churners than average.", {"size": 13, "bullet": True}),
                                  (f"Accuracy at 0.5 cut-off is {pct(R['final_acc_holdout'],1)} vs {pct(R['baseline_acc_holdout'],1)} for the baseline - shown for the tutorial's metric, but AUC is the metric we trust.", {"size": 13, "bullet": True})], spacing=4)
add_text(s, 7.0, 4.65, 5.9, 2.3, [("Honest limits", {"size": 15, "bold": True, "color": RED}),
                                  ("Behavioural data alone gives a moderate ceiling: published AUCs are ~0.62 on one public carrier dataset [18] and ~0.76 for random forests averaged over eleven telecom datasets [17]; ours is in that range.", {"size": 13, "bullet": True}),
                                  ("Hence the proposal does **not** rely on a perfect model: it contacts only the few customers where the numbers clearly pay.", {"size": 13, "bullet": True})], spacing=4)

# ================================================================= 10. drivers
s = new_slide("What drives churn: three of the top drivers are actionable timing levers", "Own analysis (permutation importance and partial dependence on hold-out); [11] handset upgrade effect")
add_img(s, "fig_importance.png", 0.35, 1.5, 6.3, 4.3)
add_img(s, "fig_pdp.png", 6.7, 1.5, 6.3, 2.2)
add_text(s, 6.8, 3.8, 6.1, 3.1, [("From driver to lever", {"size": 15, "bold": True, "color": NAVY}),
                                 ("**Tenure and handset age** (can be acted on): trigger an anniversary check-in ~1 month before month 10, and an upgrade offer before the handset reaches ~300 days.", {"size": 12.5, "bullet": True}),
                                 ("**Falling usage** (early warning): score refreshes monthly; a drop in minutes moves a customer up the list.", {"size": 12.5, "bullet": True}),
                                 ("**Plan price, credit class, region**: segment and budget rules, not individual actions.", {"size": 12.5, "bullet": True}),
                                 ("Not actionable: demographics - used as controls only.", {"size": 12.5, "bullet": True, "color": GREY})], spacing=3)
add_text(s, 0.5, 5.9, 6.2, 1.1, [("Robustness: the same top drivers appear in gain-based and permutation importance, and in the simple EDA charts - one story, three views.", {"size": 12.5, "color": INK})])

# ================================================================= 11. business model
s = new_slide("Business model: contact a customer only if the expected gain beats the cost", "[8] Verbraken et al. 2013; [10] cost-matrix formulation; [3][4][5] churn; [6][7] cost ratios; own assumptions")
add_text(s, 0.5, 1.55, 6.3, 1.0, [("Expected gain of contacting customer i  =", {"size": 15, "bold": True, "color": NAVY}),
                                  ("P(churn) x [ P(accepts) x (value kept - offer) - contact cost ]  -  P(stays) x [ P(claims anyway) x offer + contact cost ]", {"size": 13.5, "color": INK})], spacing=2)
add_text(s, 0.5, 2.85, 6.3, 3.9, [("How the model enters", {"size": 15, "bold": True, "color": NAVY}),
                                  ("Churn probability is **calibrated** and re-weighted from the 50% sample to a realistic 2% monthly rate with Bayes' rule.", {"size": 13, "bullet": True}),
                                  ("Customer value is **individual**: 24 months x 30% margin x the customer's own monthly revenue, so a high-revenue customer justifies a larger list than a low-revenue one.", {"size": 13, "bullet": True}),
                                  ("Customers are contacted if, and only if, expected gain > 0. No threshold to tune, and it re-optimises when costs change.", {"size": 13, "bullet": True}),
                                  ("Uncertainty is handled openly: take-up and leakage are the two numbers the pilot must measure.", {"size": 13, "bullet": True, "color": RED})], spacing=4)
rows = [["Assumption", "Base", "Basis"],
        ["True monthly churn", "2%", "Over-sampled sample; carriers report 0.9-1.0% postpaid, 2.7-3.7% prepaid [3][4][5][9]"],
        ["Churner accepts offer (gamma)", "30%", "Literature prior mean [8]"],
        ["Loyal customer claims offer (alpha)", "30%", "Assumed; literature worst case 100% - see sensitivity"],
        ["Value of a saved customer", f"${R['clv_mean']:,.0f}", "24 months x 30% margin x own revenue; AT&T service EBITDA margin is 57% [5]"],
        ["Offer / contact cost", "$10 / $1", "Literature defaults [8][10] (credit; SMS + e-mail)"]]
table(s, 6.95, 1.6, 5.95, 4.3, rows, col_w=[2.0, 0.75, 3.2], size=11)
add_text(s, 6.95, 6.0, 5.95, 0.95, [("All of these are inputs, not results: slide 13 shows what happens when they are wrong.", {"size": 12.5, "color": GREY})])

# ================================================================= 12. impact
s = new_slide(f"Impact: targeting by expected profit nets {usd(profit_m)} a month per 100,000 customers, while contacting everyone loses {usd(-EV['profit'])}", "Own analysis (hold-out, base case); 90% bootstrap interval; assumptions on slide 11")
add_img(s, "fig_strategies.png", 0.35, 1.5, 6.8, 4.1)
lo, hi = R["profit_base_ci"]
rows = [["Per 100,000 customers / month", "Value-aware model"],
        ["Customers contacted", f"{V['contacts']:,.0f} ({pct(V['contacts']/100000,1)})"],
        ["Churners reached (of ~2,000)", f"{V['churners_reached']:,.0f}"],
        ["Customers saved (30% accept)", f"{V['saved']:,.0f}"],
        ["Campaign spend", f"${V['spend']:,.0f}"],
        ["Net profit", f"${V['profit']:,.0f}"],
        ["ROI on spend", f"{V['roi']*100:,.0f}%"],
        ["90% interval for profit", f"${lo:,.0f} to ${hi:,.0f}"]]
table(s, 7.4, 1.6, 5.5, 3.2, rows, col_w=[3.1, 2.4], size=12.5, hl_row=5)
add_text(s, 7.4, 4.95, 5.5, 2.0, [("Reading it", {"size": 15, "bold": True, "color": NAVY}),
                                  (f"Same offer, same costs: ordering by **risk only** earns {usd(RK['profit'])}; the **value-aware** list earns {usd(V['profit'])}.", {"size": 13, "bullet": True}),
                                  (f"Contacting everyone in a lifecycle window (the rule of thumb, {RU['contacts']:,.0f} contacts) loses {usd(-RU['profit'])} a month.", {"size": 13, "bullet": True}),
                                  (f"Annualised, base case: about **{usd(profit_y)}** per 100,000 customers (assumes monthly cohorts and a stable save rate).", {"size": 13, "bullet": True})], spacing=3)
add_text(s, 0.5, 5.75, 6.6, 1.2, [("Why not simply contact everyone in the danger windows? The 10-month and 300-day windows raise risk, but at a realistic 2% churn most contacted customers would stay, and every one who claims an offer costs money. The 90% interval reflects hold-out sampling noise only, not uncertainty in the assumptions.", {"size": 12, "color": INK})])

# ================================================================= 13. sensitivity + risks
s = new_slide("Sensitivity and risks: offer design decides whether the campaign pays", "Own analysis; profits in $ thousand per month per 100,000 customers; boxed cell = base case")
add_img(s, "fig_sens_alpha_gamma.png", 0.35, 1.45, 6.3, 3.9)
add_img(s, "fig_sens_pi_clv.png", 6.7, 1.45, 6.3, 3.9)
ps, bs, us = sc["Pessimistic"], sc["Base"], sc["Upside"]
rows = [["Scenario", "Monthly churn", "Churner accepts", "Loyal claims offer", "Net profit / month"],
        ["Pessimistic", pct(ps['params']['pi'], 1), pct(ps['params']['gamma']), pct(ps['params']['alpha']), f"${ps['profit']:,.0f}"],
        ["Base", pct(bs['params']['pi'], 1), pct(bs['params']['gamma']), pct(bs['params']['alpha']), f"${bs['profit']:,.0f}"],
        ["Upside", pct(us['params']['pi'], 1), pct(us['params']['gamma']), pct(us['params']['alpha']), f"${us['profit']:,.0f}"]]
table(s, 0.5, 5.45, 7.0, 1.45, rows, col_w=[1.5, 1.2, 1.4, 1.5, 1.4], size=11.5, hl_row=2)
add_text(s, 7.7, 5.4, 5.2, 1.6, [("Key risks", {"size": 14, "bold": True, "color": RED}),
                                 (f"Pessimistic case is only break-even: about {ps['contacts']:.0f} customers per 100,000 are worth contacting.", {"size": 12, "bullet": True}),
                                 ("Offer leakage and weak save rates (measure first).", {"size": 12, "bullet": True}),
                                 ("Correlation is not causation: lifecycle windows may be contract effects; use a randomised hold-out.", {"size": 12, "bullet": True}),
                                 ("Data age and the 2% rate assumption; recalibrate on live data.", {"size": 12, "bullet": True})], spacing=2)

# ================================================================= 14. proposal & roadmap
s = new_slide("Proposal: a lifecycle retention engine, proven in an 8-week pilot before any scale-up", "Own proposal; [8] profit-driven churn framework; [11][12] on upgrades and renewal windows")
cols = [("Phase 1: Pilot (weeks 1-8)", NAVY, ["Score the base monthly with the model; build the contact list by expected gain.", "Randomise: 50% of the list gets the offer and 50% is held out, so we can measure the **true save rate** and **leakage**.", "Two plays: an anniversary check-in (month 9) for customers still on their original handset, and an upgrade offer before the handset reaches ~300 days.", "Cost: model run + $1 per contact + $10 per accepted offer."]),
        ("Phase 2: Scale (months 3-6)", TEAL, ["Replace assumptions with the measured save rate and leakage; re-optimise the list automatically.", "Add channels (app, store, call) with their own costs.", "Refresh the scores monthly."]),
        ("Phase 3: Optimise (month 6+)", RED, ["Uplift models: target customers who respond **because of** the offer, not merely likely churners.", "Test other levers: plan change, equipment financing.", "Quarterly drift review."])]
for i, (ttl, colr, items) in enumerate(cols):
    x = 0.5 + i * 4.18
    box(s, x, 1.6, 3.95, 0.7, ttl, fill=colr, size=16)
    add_text(s, x, 2.4, 3.95, 3.0, [(t, {"bullet": True}) for t in items], size=12.5, spacing=5)
card(s, 0.5, 5.5, 3.95, 1.35, "KPIs", "Saves per 1,000 contacts, offer leakage, net profit per 100k customers, churn in treated vs hold-out", NAVY, big_size=18, small_size=11.5)
card(s, 4.68, 5.5, 3.95, 1.35, "Go / no-go", f"Scale only if measured profit stays above $0 at the pilot's own take-up numbers (base case {usd(profit_m)} a month per 100,000 customers)", TEAL, big_size=18, small_size=11.5)
card(s, 8.86, 5.5, 3.95, 1.35, "Budget at risk", f"At base case about {usd(V['spend'])} a month per 100,000 customers in campaign spend", RED, big_size=18, small_size=11.5)

# ================================================================= 15. references
s = new_slide("References", "")
refs = [
    "[1] CTIA (2025). 2025 Annual Survey Highlights. https://api.ctia.org/wp-content/uploads/2025/09/2025-Annual-Survey-Highlights.pdf",
    "[2] Light Reading (2026). Inflation mostly skipped the wireless bill (CTIA data, 2025 connections). https://www.lightreading.com/wireless/inflation-mostly-skipped-the-wireless-bill",
    "[3] T-Mobile US (2026). Q4 2025 results (postpaid phone and prepaid churn). https://investor.t-mobile.com/static-files/0e19010d-e388-4de7-81c0-f9bb87c7e4fe",
    "[4] Verizon (2026). Financial and Operating Information, 4Q25. https://www.verizon.com/about/sites/default/files/2026-02/vz_4q25_foi_013026r.pdf",
    "[5] AT&T (2025). Third-Quarter 2025 results. https://investors.att.com/~/media/Files/A/ATT-IR-V2/financial-reports/quarterly-earnings/2025/3Q-2025/3Q25_ATT_News_Release.pdf",
    "[6] Gallo, A. (2014). The Value of Keeping the Right Customers. Harvard Business Review. https://hbr.org/2014/10/the-value-of-keeping-the-right-customers",
    "[7] Customer Acquisition and Retention Spending: An Analytical Model and Empirical Investigation in Wireless Telecommunications Markets. Journal of Marketing Research. https://doi.org/10.1509/jmr.14.0170",
    "[8] Verbraken, T., Verbeke, W., Baesens, B. (2013). A Novel Profit Maximizing Metric for Measuring Classification Performance of Customer Churn Prediction Models. IEEE TKDE 25(5). https://doi.org/10.1109/TKDE.2012.50",
    "[9] Duke University Teradata Center for CRM. The Churn Game (case describing a 50% over-sampled calibration set and ~2% monthly churn). https://www.yumpu.com/en/document/view/8583164/the-churn-game-duke-universitys-fuqua-school-of-business",
    "[10] empulse documentation (2026). Cell2Cell customer churn: cost matrix and benchmark context. https://empulse.readthedocs.io/en/stable/guide/datasets/cell2cell.html",
    "[11] Using Cox regression to model customer time to churn in the wireless telecommunications industry. J. of Targeting, Measurement and Analysis for Marketing (2011). https://doi.org/10.1057/jt.2011.1",
    "[12] Benzinga (2026). T-Mobile, Verizon, AT&T Go All-In On Discounts As Churn Surge Hits. https://www.benzinga.com/markets/tech/26/03/51348413/t-mobile-verizon-att-go-all-in-on-discounts-as-churn-surge-hits",
    "[13] Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS.",
    "[14] Chen, T., Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.",
    "[15] Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research 12.",
    "[16] Matsuo-Iwasawa Lab (2026). GCI World 2026 Winter Final Assignment: README, Company A Dataset Overview, tutorial.ipynb; course lecture materials.",
    "[17] An empirical comparison of techniques for the class imbalance problem in churn prediction (11 data sets). University of Southampton ePrints. https://eprints.soton.ac.uk/413694/1/An_empirical_comparison_of_techniques_for_the_class_imbalance_problem_in_churn_prediction.pdf",
    "[18] Customer Churn Prediction on the Cell2Cell Dataset by Deep Learning (thesis), Tilburg University. http://arno.uvt.nl/show.cgi?fid=160769",
]
add_text(s, 0.5, 1.5, 12.3, 5.5, [(r, {"size": 11.5, "space": 4}) for r in refs])

out = Path(f"{ACCOUNT}.pptx"); prs.save(out); print("saved", out, "slides:", len(prs.slides))
