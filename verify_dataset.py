"""Ad-hoc verifier for the baked-in narratives + hard invariants. Read-only."""
import csv, statistics, subprocess, sys, collections
import generate_dataset as G   # to assert encoded differentials at source level

D = "data/"

def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

ads = load(D + "ads.csv")
eng = load(D + "engagement.csv")
meta = {a["ad_id"]: a for a in ads}

def fnum(s):
    return float(s) if s not in ("", None) else None

fails, checks = [], 0
def chk(name, cond, extra=""):
    global checks
    checks += 1
    if not cond:
        fails.append(f"FAIL: {name}  {extra}")

# group engagement rows by ad and attach theme/format/placement/day index
by_ad = collections.defaultdict(list)
for r in eng:
    by_ad[r["ad_id"]].append(r)
for aid, rows in by_ad.items():
    rows.sort(key=lambda r: r["date"])

# ---------------------------------------------------------------
# 1. HARD INVARIANTS (must hold on every applicable row)
# ---------------------------------------------------------------
for r in eng:
    a = meta[r["ad_id"]]
    fmt = a["format"]
    imp = int(r["impressions"]); clk = int(r["clicks"])
    chk("impressions>=1", imp >= 1)
    chk("clicks>=1 (no div-by-zero)", clk >= 1, r["ad_id"]+" "+r["date"])
    if fmt == "video":
        p25 = int(r["video_plays_25pct"]); p75 = int(r["video_plays_75pct"])
        apt = fnum(r["avg_play_time_sec"]); vlen = float(a["video_length_sec"])
        chk("p75 < p25", p75 < p25, f"{r['ad_id']} {r['date']} {p75}!<{p25}")
        chk("p25 < impressions", p25 < imp, f"{r['ad_id']} {p25}!<{imp}")
        chk("avg_play_time <= video_length", apt <= vlen, f"{r['ad_id']} {apt}>{vlen}")
    else:
        chk("video cols null for non-video",
            r["video_plays_25pct"] == "" and r["video_plays_75pct"] == ""
            and r["avg_play_time_sec"] == "")
    # carousel-only cols
    if fmt == "carousel":
        chk("carousel has card metrics", r["card_swipe_rate"] != "")
    else:
        chk("card metrics null off-carousel",
            r["card_swipe_rate"] == "" and r["last_card_reach_rate"] == "")
    # meta social cols: only meta static/carousel
    social_present = r["likes"] != ""
    should = (a["platform"] == "meta" and fmt in ("static", "carousel"))
    chk("social presence matches meta static/carousel", social_present == should,
        f"{r['ad_id']} {a['platform']}/{fmt} present={social_present}")

# ---------------------------------------------------------------
# 2. PLACEMENT DISTRIBUTION (Fix 2)
# ---------------------------------------------------------------
ALLOWED = {
    "wonder": {"feed", "reels", "youtube"},
    "safety": {"feed", "stories"},
    "confidence": {"feed", "stories", "search"},
    "connection": {"feed", "reels", "stories"},
}
theme_placements = collections.defaultdict(set)
for a in ads:
    theme_placements[a["theme"]].add(a["placement"])
for th, allowed in ALLOWED.items():
    chk(f"{th} placements within allowed set", theme_placements[th] <= allowed,
        f"got {theme_placements[th]}")
chk("reels ABSENT from safety", "reels" not in theme_placements["safety"])
chk("youtube only in wonder",
    all("youtube" not in theme_placements[t] for t in ("safety","confidence","connection")))
chk("search only in confidence",
    all("search" not in theme_placements[t] for t in ("wonder","safety","connection")))

# ---------------------------------------------------------------
# helpers to compute mean realized metric over a filter
# ---------------------------------------------------------------
def rows_where(pred):
    return [r for r in eng if pred(r, meta[r["ad_id"]])]

def mean_metric(rows, col):
    vals = [fnum(r[col]) for r in rows if fnum(r[col]) is not None]
    return statistics.mean(vals) if vals else None

# ---------------------------------------------------------------
# 3. FORMAT-LEVEL DIFFERENTIALS (Fix 3)
#    Encoded as constant multipliers (stable across themes/days) -> verify the
#    SOURCE constants directly, since the 24 unique-combo ads make realized
#    global averages confounded by placement/platform/the connection outliers.
# ---------------------------------------------------------------
chk("FORMAT_CTR_MULT video highest", G.FORMAT_CTR_MULT["video"] > G.FORMAT_CTR_MULT["static"]
    > G.FORMAT_CTR_MULT["carousel"], str(G.FORMAT_CTR_MULT))
chk("FORMAT_CVR_MULT carousel highest, static lowest",
    G.FORMAT_CVR_MULT["carousel"] > G.FORMAT_CVR_MULT["video"] > G.FORMAT_CVR_MULT["static"],
    str(G.FORMAT_CVR_MULT))
# Robust realized checks (survive confounds): video highest CTR, carousel lowest
ctr_by_fmt = {f: mean_metric(rows_where(lambda r,a: a["format"]==f), "CTR")
              for f in ("video","static","carousel")}
chk("realized: video highest CTR", ctr_by_fmt["video"] > ctr_by_fmt["static"]
    and ctr_by_fmt["video"] > ctr_by_fmt["carousel"], str(ctr_by_fmt))
chk("realized: carousel lowest CTR", ctr_by_fmt["carousel"] < ctr_by_fmt["static"], str(ctr_by_fmt))
cvr_by_fmt = {f: mean_metric(rows_where(lambda r,a: a["format"]==f), "CVR")
              for f in ("video","static","carousel")}
chk("realized: carousel CVR > static CVR (format gap survives)",
    cvr_by_fmt["carousel"] > cvr_by_fmt["static"], str(cvr_by_fmt))
# Confound-free realized slice: within CONFIDENCE (no outliers), carousel>video CVR
conf_cvr_fmt = {f: mean_metric(rows_where(lambda r,a: a["theme"]=="confidence" and a["format"]==f), "CVR")
                for f in ("video","static","carousel")}
chk("realized (confidence): carousel CVR > video CVR > static CVR",
    conf_cvr_fmt["carousel"] > conf_cvr_fmt["video"] > conf_cvr_fmt["static"], str(conf_cvr_fmt))

# ---------------------------------------------------------------
# 4. WONDER DIMINISHING RETURNS (Fix 1)
#    Narrative: CPI improves steeply (warmup), then slower (scaling), BOTTOMS
#    near day 60, then rises 8-12% off that floor (mature ceiling). Compare the
#    floor window to the tail -- NOT phase means (which hide the floor).
# ---------------------------------------------------------------
def day_index(r):
    return by_ad[r["ad_id"]].index(r)
wonder = [r for r in eng if meta[r["ad_id"]]["theme"]=="wonder"]
def win_cpi(lo, hi):
    return mean_metric([r for r in wonder if lo <= day_index(r) < hi], "CPI")
cpi_early  = win_cpi(0, 10)    # opening
cpi_warm   = win_cpi(25, 35)   # end of warmup
cpi_floor  = win_cpi(55, 65)   # the efficiency floor (~day 60)
cpi_tail   = win_cpi(80, 90)   # mature tail
chk("wonder CPI improves through warmup", cpi_warm < cpi_early,
    f"early={cpi_early:.3f} warm={cpi_warm:.3f}")
chk("wonder CPI keeps improving into the floor", cpi_floor < cpi_warm,
    f"warm={cpi_warm:.3f} floor={cpi_floor:.3f}")
chk("wonder diminishing returns (warmup gain > scaling gain)",
    (cpi_early - cpi_warm) > (cpi_warm - cpi_floor),
    f"warmup_gain={cpi_early-cpi_warm:.3f} scaling_gain={cpi_warm-cpi_floor:.3f}")
chk("wonder mature CPI rises above the floor", cpi_tail > cpi_floor,
    f"floor={cpi_floor:.3f} tail={cpi_tail:.3f}")
worsen = (cpi_tail - cpi_floor) / cpi_floor * 100
chk("wonder mature worsening in 5-18% band", 5 <= worsen <= 18, f"{worsen:.1f}%")

# Spend-DRIVEN, not just time-driven: within each Wonder ad (placement/platform/
# quality all held constant) higher daily spend must come with higher CPI in the
# mature window -> the spend-vs-CPI relationship is a curve. Verify per-ad
# correlation, which removes the cross-ad confound that a pooled mean suffers.
mature_days = lambda aid: [r for r in by_ad[aid] if by_ad[aid].index(r) >= 55]
def pearson(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    den = (sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))**0.5
    return num/den if den else 0.0
wonder_ids = [a["ad_id"] for a in ads if a["theme"]=="wonder"]
corrs = []
for aid in wonder_ids:
    rows = mature_days(aid)
    sp = [float(r["spend"]) for r in rows]
    cp = [fnum(r["CPI"]) for r in rows if fnum(r["CPI"]) is not None]
    if len(cp) == len(sp) and len(sp) > 5:
        corrs.append(pearson(sp, cp))
chk("within-ad spend<->CPI positive on average (curve, not line)",
    statistics.mean(corrs) > 0.05, f"mean_corr={statistics.mean(corrs):.3f}")
threshold_crossers = sum(1 for r in wonder if float(r["spend"]) > 190)
chk("saturation threshold actually exercised (>$190 days exist)",
    threshold_crossers > 30, f"n={threshold_crossers}")
# unit-test the penalty function shape
chk("penalty: no drag below threshold", G.spend_saturation_penalty("wonder", 150) == 1.0)
chk("penalty: drag above threshold", G.spend_saturation_penalty("wonder", 260) < 1.0)
chk("penalty: never applies to other themes",
    G.spend_saturation_penalty("safety", 999) == 1.0)

# diminishing returns must NOT leak: other themes' spend stays in original band
for th in ("safety","confidence","connection"):
    mx = max(float(r["spend"]) for r in eng if meta[r["ad_id"]]["theme"]==th)
    chk(f"{th} spend NOT scaled up (<150)", mx < 150, f"max={mx:.1f}")

# ---------------------------------------------------------------
# 5. PLACEMENT CPI DIFFERENTIALS (Fix 2)
#    Encoded as constant per-placement multipliers. The implied CPI multiplier
#    is CPM_MULT / (CTR_MULT * CVR_MULT). Verify the encoded ordering at source
#    (realized per-placement means are confounded by the format mix on each
#    placement, e.g. Safety's stories are carousels, its feed are statics).
# ---------------------------------------------------------------
def place_cpi_index(p):
    return G.PLACEMENT_CPM_MULT[p] / (G.PLACEMENT_CTR_MULT[p] * G.PLACEMENT_CVR_MULT[p])
feed_idx = place_cpi_index("feed")
chk("encoded reels CPI ~15-20% below feed",
    0.80 <= place_cpi_index("reels")/feed_idx <= 0.85,
    f"ratio={place_cpi_index('reels')/feed_idx:.3f}")
chk("encoded stories CPI ~10% above feed",
    1.07 <= place_cpi_index("stories")/feed_idx <= 1.13,
    f"ratio={place_cpi_index('stories')/feed_idx:.3f}")
chk("encoded youtube CPI ~12% below feed",
    0.86 <= place_cpi_index("youtube")/feed_idx <= 0.90,
    f"ratio={place_cpi_index('youtube')/feed_idx:.3f}")
chk("encoded search CPI is highest of all placements",
    place_cpi_index("search") == max(place_cpi_index(p) for p in G.PLACEMENT_CPM_MULT),
    f"search={place_cpi_index('search'):.3f}")
chk("encoded search CVR_MULT is highest", G.PLACEMENT_CVR_MULT["search"]
    == max(G.PLACEMENT_CVR_MULT.values()), str(G.PLACEMENT_CVR_MULT))

# Confound-free realized checks: reels is all-video in wonder/connection, so its
# strong CPI shows cleanly. This is the actual cross-theme signal the agent uses.
def theme_place_cpi(theme, place):
    return mean_metric(rows_where(lambda r,a: a["theme"]==theme and a["placement"]==place), "CPI")
w_feed = theme_place_cpi("wonder","feed")
chk("realized: wonder reels CPI < feed", theme_place_cpi("wonder","reels") < w_feed,
    f"reels={theme_place_cpi('wonder','reels'):.3f} feed={w_feed:.3f}")
chk("realized: wonder youtube CPI < feed", theme_place_cpi("wonder","youtube") < w_feed,
    f"yt={theme_place_cpi('wonder','youtube'):.3f} feed={w_feed:.3f}")
chk("realized: connection reels CPI < feed",
    theme_place_cpi("connection","reels") < theme_place_cpi("connection","feed"),
    f"reels={theme_place_cpi('connection','reels'):.3f} feed={theme_place_cpi('connection','feed'):.3f}")
# search shows highest CVR among confidence placements (high-intent clicks)
conf_cvr = {p: mean_metric(rows_where(lambda r,a: a["theme"]=="confidence" and a["placement"]==p),"CVR")
            for p in ("feed","stories","search")}
chk("realized: confidence search highest CVR", conf_cvr["search"]==max(conf_cvr.values()), str(conf_cvr))

# ---------------------------------------------------------------
# 6. CONNECTION OUTLIERS (Fix 3) -- two video ads, high CVR, low variance
# ---------------------------------------------------------------
conn_ads = [a for a in ads if a["theme"]=="connection"]
def ad_cvr_mean(aid): return mean_metric(by_ad[aid], "CVR")
def ad_ctr_cv(aid):  # coefficient of variation of CTR (volatility proxy)
    vals=[fnum(r["CTR"]) for r in by_ad[aid]]
    return statistics.pstdev(vals)/statistics.mean(vals)
video_ads = [a["ad_id"] for a in conn_ads if a["format"]=="video"]
other_ads = [a["ad_id"] for a in conn_ads if a["format"]!="video"]
chk("exactly 2 connection video ads", len(video_ads)==2, str(video_ads))
min_out_cvr = min(ad_cvr_mean(a) for a in video_ads)
max_oth_cvr = max(ad_cvr_mean(a) for a in other_ads)
chk("connection outliers have highest CVR", min_out_cvr > max_oth_cvr,
    f"out_min={min_out_cvr:.3f} other_max={max_oth_cvr:.3f}")
max_out_cv = max(ad_ctr_cv(a) for a in video_ads)
min_oth_cv = min(ad_ctr_cv(a) for a in other_ads)
chk("connection outliers least volatile (CTR CV)", max_out_cv < min_oth_cv,
    f"out_max_cv={max_out_cv:.3f} other_min_cv={min_oth_cv:.3f}")

# ---------------------------------------------------------------
# 7. SAFETY DECLINE PRESERVED (video depth drops after day 35)
# ---------------------------------------------------------------
safety_vid = [a["ad_id"] for a in ads if a["theme"]=="safety" and a["format"]=="video"]
def safety_phase(col, lo, hi):
    rows=[r for aid in safety_vid for r in by_ad[aid] if lo<=by_ad[aid].index(r)<hi]
    return mean_metric(rows, col)
for col in ("hold_rate","video_plays_75pct","avg_play_time_sec"):
    early=safety_phase(col,0,35); late=safety_phase(col,55,90)
    chk(f"safety {col} drops after day 35", late < early*0.85,
        f"early={early:.3f} late={late:.3f}")
# 25pct stays relatively stable (not a sharp drop)
e25=safety_phase("video_plays_25pct",0,35); l25=safety_phase("video_plays_25pct",55,90)
chk("safety video_plays_25pct relatively stable", l25 > e25*0.75,
    f"early={e25:.1f} late={l25:.1f}")

# ---------------------------------------------------------------
# 8. REPRODUCIBILITY
# ---------------------------------------------------------------
import hashlib
def md5(path):
    return hashlib.md5(open(path,"rb").read()).hexdigest()
h1e, h1a = md5(D+"engagement.csv"), md5(D+"ads.csv")
subprocess.run([sys.executable,"../generate_dataset.py"], cwd=D, check=True,
               stdout=subprocess.DEVNULL)
chk("engagement.csv reproducible", md5(D+"engagement.csv")==h1e)
chk("ads.csv reproducible", md5(D+"ads.csv")==h1a)

# ---------------------------------------------------------------
print(f"\nRan {checks} checks.")
if fails:
    print(f"{len(fails)} FAILURES:")
    for f in fails[:40]:
        print(" ", f)
    sys.exit(1)
print("ALL CHECKS PASSED")
print(f"\n-- summary numbers --")
print(f"realized format CTR: " + ", ".join(f"{k}={v:.4f}" for k,v in ctr_by_fmt.items()))
print(f"realized format CVR: " + ", ".join(f"{k}={v:.4f}" for k,v in cvr_by_fmt.items()))
print(f"wonder CPI: early={cpi_early:.3f} warmup={cpi_warm:.3f} floor={cpi_floor:.3f} "
      f"tail={cpi_tail:.3f} (tail +{worsen:.1f}% off floor)")
print(f"wonder within-ad mean spend<->CPI corr: {statistics.mean(corrs):.3f}")
print(f"wonder realized reels/feed/youtube CPI: {theme_place_cpi('wonder','reels'):.3f}"
      f"/{w_feed:.3f}/{theme_place_cpi('wonder','youtube'):.3f}")
print(f"connection outlier CVR (min) {min_out_cvr:.3f} vs other (max) {max_oth_cvr:.3f}")
