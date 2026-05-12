import os

with open('control_panel/models.py', 'a', encoding='utf-8') as f:
    f.write("""

class TradingReport(models.Model):
    REPORT_TYPES = [
        ('DAILY', 'Daily Summary'),
        ('WEEKLY', 'Weekly Rolling Analysis'),
        ('MONTHLY', 'Monthly Performance'),
    ]
    report_type = models.CharField(max_length=15, choices=REPORT_TYPES)
    timestamp = models.DateTimeField(auto_now_add=True)
    markdown_path = models.CharField(max_length=255)
    pdf_path = models.CharField(max_length=255, null=True, blank=True)
    total_revenue = models.DecimalField(max_digits=15, decimal_places=2, default=0.0)
    total_trades = models.IntegerField(default=0)
    win_rate = models.FloatField(default=0.0)

    def __str__(self):
        return f"{self.report_type} Report - {self.timestamp.strftime('%Y-%m-%d')}"
""")
print("Successfully appended TradingReport to models.py")
