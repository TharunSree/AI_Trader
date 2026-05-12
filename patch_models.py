with open('control_panel/models.py', 'r', encoding='utf-8') as f:
    text = f.read()

broker_class = '''
class BrokerAccount(models.Model):
    name = models.CharField(max_length=100, help_text="e.g. 'Main Alpaca Account'")
    api_key = models.CharField(max_length=255)
    secret_key = models.CharField(max_length=255)
    base_url = models.CharField(max_length=255, default='https://paper-api.alpaca.markets')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class PaperTrader(models.Model):'''

text = text.replace("class PaperTrader(models.Model):", broker_class)

paper_trader_fields_target = "    is_live = models.BooleanField(default=False)"
paper_trader_fields_replacement = '''    is_live = models.BooleanField(default=False)
    account = models.ForeignKey(BrokerAccount, on_delete=models.SET_NULL, null=True, blank=True, related_name='instances')
    live_net_profit = models.FloatField(default=0.0)'''

text = text.replace(paper_trader_fields_target, paper_trader_fields_replacement)

with open('control_panel/models.py', 'w', encoding='utf-8') as f:
    f.write(text)
