with open('templates/papertrading_fleet.html', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''        .then((data) => {
          document.getElementById('shared-equity').textContent = fmtCurrency(data.equity);
          document.getElementById('shared-buying-power').textContent = fmtCurrency(data.buying_power);
          document.getElementById('shared-bp').textContent = fmtCurrency(data.buying_power);'''

replacement = '''        .then((data) => {
          const eqOpts = document.getElementById('shared-equity'); if(eqOpts) eqOpts.textContent = fmtCurrency(data.equity);
          const bpOpts = document.getElementById('shared-buying-power'); if(bpOpts) bpOpts.textContent = fmtCurrency(data.buying_power);
          const bp2Opts = document.getElementById('shared-bp'); if(bp2Opts) bp2Opts.textContent = fmtCurrency(data.buying_power);'''

if target in text:
    text = text.replace(target, replacement)
    with open('templates/papertrading_fleet.html', 'w', encoding='utf-8') as f:
        f.write(text)
