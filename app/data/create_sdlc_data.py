import os, json, random, math
from datetime import date
from datetime import datetime
from dateutil.relativedelta import relativedelta

random.seed(42)

num_apps = 30
controls = [f"SDLC{i}" for i in range(1,25)]
include_current = True
today = date(2025, 10, 2)

def last_n_month_starts(n, end_date, include_current=True):
    anchor = date(end_date.year, end_date.month, 1)
    if not include_current:
        anchor = (anchor - relativedelta(months=1))
    months = []
    for i in range(n):
        d = anchor - relativedelta(months=(n - 1 - i))
        months.append(d)
    return months

months = last_n_month_starts(6, today, include_current=include_current)

itso_first = ["Aiden","Bella","Caleb","Dylan","Eva","Felix","Grace","Henry","Iris","Jack",
              "Liam","Mia","Noah","Olivia","Ethan","Chloe","Logan","Sofia","Mason","Layla",
              "Zoe","Lucas","Amelia","Wyatt","Nora","Ezra","Harper","Levi","Maya"]
itso_last  = ["Brooks","Carter","Foster","Hayes","Johnson","Martin","Nolan","Owens","Patel","Quinn",
              "Reed","Steele","Turner","Vargas","Wong","Young","Zimmer","Blake","Cole","Diaz",
              "Ellis","Floyd","Gates","Hicks","Irwin","James","Klein","Lane","Moore","Ng"]

def gen_app_id(idx):
    base = f"APP{idx:02d}"
    suffix_pool = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    random.seed(idx)
    suffix = "".join(random.choice(suffix_pool) for _ in range(9 - len(base)))
    return base + suffix

apps = [gen_app_id(i) for i in range(1, num_apps+1)]

itso_names = {}
for i, app in enumerate(apps):
    itso_names[app] = f"{itso_first[i % len(itso_first)]} {itso_last[i % len(itso_last)]}"

out_dir = "/Users/abhishekgururani/ProjectML/app/data"
os.makedirs(out_dir, exist_ok=True)

app_base_risk = {app: random.uniform(0.05, 0.35) for app in apps}
control_weight = {c: random.uniform(0.8, 1.2) for c in controls}

def control_value(prob):
    return 1 if random.random() < prob else 0

for m in months:
    month_tag = m.strftime("%Y%m")
    file_path = os.path.join(out_dir, f"SDLC_{month_tag}.json")
    records = []
    for app in apps:
        rec = {
            "applicationID": app,
            "Month": month_tag,
            "ITSO_Name": itso_names[app]
        }
        month_factor = 1.0 + 0.02 * ((m.year - months[0].year) * 12 + (m.month - months[0].month))
        base = app_base_risk[app] * month_factor
        gaps = 0
        for c in controls:
            p = min(0.6, base * control_weight[c])
            v = control_value(p)
            rec[c] = v
            gaps += v
        fail_prob = min(0.9, 0.1 + 0.15 * gaps)
        rec["fail_flag"] = 1 if random.random() < fail_prob else 0
        records.append(rec)
    with open(file_path, "w") as f:
        json.dump(records, f, indent=2)

print(f"Wrote {len(months)} files to {out_dir}")
