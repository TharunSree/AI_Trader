import xml.etree.ElementTree as ET
import urllib.request
import urllib.parse
import json
import re
from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import datetime, timedelta
from django.core.mail import send_mail
from django.conf import settings
from django.db import models
from control_panel.models import Game, GameBetaInfo, WatchlistGame, BudgetWatchlistGame

class Command(BaseCommand):
    help = "Runs background checks for game betas, watchlist updates, and discount prices"

    def handle(self, *args, **options):
        self.stdout.write("Initializing Arcade Lounge monitors...")
        self.check_beta_recruitment()
        self.check_watchlist_releases()
        self.check_discount_sales()
        self.stdout.write("All monitors execution completed successfully.")

    def fetch_rss_feed(self, url):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=8) as res:
                xml_data = res.read()
            return ET.fromstring(xml_data)
        except Exception as e:
            self.stderr.write(f"Failed to fetch feed {url}: {e}")
            return None

    def send_email_notification(self, subject, body, html_message=None):
        recipient = getattr(settings, 'EMAIL_HOST_USER', None)
        if not recipient:
            self.stdout.write("Email recipient not configured. Skipping alert dispatch.")
            return
        try:
            send_mail(
                subject=subject,
                message=body,
                from_email=recipient,
                recipient_list=[recipient],
                html_message=html_message,
                fail_silently=False
            )
            self.stdout.write(f"Dispatched email notification: {subject.encode('ascii', errors='replace').decode('ascii')}")
        except Exception as e:
            self.stderr.write(f"Failed to dispatch email: {str(e).encode('ascii', errors='replace').decode('ascii')}")

    def check_beta_recruitment(self):
        self.stdout.write("Scouting beta test signups...")
        # Target games explicitly flagged for beta tracking, falling back to defaults if none are set
        target_games = Game.objects.filter(is_active=True, watch_beta_recruitment=True)
        if not target_games.exists():
            target_games = Game.objects.filter(is_active=True).filter(
                models.Q(name__icontains="genshin") | 
                models.Q(name__icontains="wuthering") | 
                models.Q(name__icontains="wuwa")
            )
        
        for game in target_games:
            query = urllib.parse.quote(f"{game.name} beta test recruitment signup when:30d")
            rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            
            root = self.fetch_rss_feed(rss_url)
            if root is None:
                continue

            items = root.findall('.//item')
            for item in items[:5]: # Check top 5 news entries
                title = item.find('title').text
                link = item.find('link').text
                
                # Check for beta keywords
                keywords = ["beta", "recruitment", "signup", "apply", "test", "registration"]
                if any(kw in title.lower() for kw in keywords):
                    # Check if already logged
                    exists = GameBetaInfo.objects.filter(game=game, signup_link=link).exists()
                    if not exists:
                        beta_info = GameBetaInfo.objects.create(
                            game=game,
                            title=title,
                            signup_link=link,
                            is_active=True
                        )
                        # Notify user
                        subject = f"🚨 BETA WATCH: {game.name} Recruitment Active!"
                        body = f"An active recruitment signup was detected for {game.name}:\n\nTitle: {title}\nLink: {link}\n\nCheck your Game Dashboard to see more details."
                        self.send_email_notification(subject, body)

    def check_watchlist_releases(self):
        self.stdout.write("Crawling upcoming watchlists...")
        watchlist = WatchlistGame.objects.all()
        for game in watchlist:
            # 1. Scaling checks frequency logic
            days_left = None
            if game.expected_release_date:
                try:
                    rel_date = datetime.strptime(game.expected_release_date, "%Y-%m-%d").date()
                    days_left = (rel_date - timezone.now().date()).days
                except ValueError:
                    pass

            interval = 7
            if days_left is not None:
                if days_left <= 30:
                    interval = 1
                elif days_left <= 90:
                    interval = 3

            # Update interval
            if game.check_interval_days != interval:
                game.check_interval_days = interval
                game.save()

            # Skip checking if last checked is within interval
            if game.last_checked_at and timezone.now() - game.last_checked_at < timedelta(days=interval):
                continue

            # 2. Query dynamic news for game
            query = urllib.parse.quote(f"{game.name} release date updates news")
            rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            root = self.fetch_rss_feed(rss_url)
            
            game.last_checked_at = timezone.now()
            game.save()

    def get_usd_to_inr_rate(self):
        try:
            req = urllib.request.Request("https://open.er-api.com/v6/latest/USD", headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as res:
                data = json.loads(res.read().decode('utf-8'))
            rate = data.get('rates', {}).get('INR', 83.5)
            self.stdout.write(f"Fetched live USD to INR exchange rate: {rate}")
            return float(rate)
        except Exception as e:
            self.stderr.write(f"Failed to fetch exchange rate, using fallback 83.5: {e}")
            return 83.5

    def check_discount_sales(self):
        self.stdout.write("Verifying budget discount prices...")
        watchlist = BudgetWatchlistGame.objects.all()
        usd_to_inr = self.get_usd_to_inr_rate()
        
        store_map = {
            "1": "Steam",
            "2": "GamersGate",
            "3": "GreenManGaming",
            "7": "Funstock",
            "11": "Humble Store",
            "15": "Fanatical",
            "18": "GOG",
            "21": "Nintendo eShop",
            "25": "Epic Games",
            "27": "Xbox Store",
            "30": "PlayStation Store"
        }

        for item in watchlist:
            encoded_title = urllib.parse.quote(item.name)
            search_url = f"https://www.cheapshark.com/api/1.0/games?title={encoded_title}"
            
            try:
                req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=8) as res:
                    search_data = json.loads(res.read().decode('utf-8'))
                
                if not search_data:
                    continue

                # Match hierarchy (Steam ID -> Exact name -> Substring -> First result)
                matched_result = None
                if item.steam_app_id:
                    for res in search_data:
                        if str(res.get('steamAppID')) == str(item.steam_app_id):
                            matched_result = res
                            break
                if not matched_result:
                    for res in search_data:
                        if res.get('external', '').lower() == item.name.lower():
                            matched_result = res
                            break
                if not matched_result:
                    for res in search_data:
                        if item.name.lower() in res.get('external', '').lower():
                            matched_result = res
                            break
                if not matched_result:
                    matched_result = search_data[0]

                game_id = matched_result.get('gameID')
                details_url = f"https://www.cheapshark.com/api/1.0/games?id={game_id}"
                
                with urllib.request.urlopen(urllib.request.Request(details_url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=8) as d_res:
                    details_data = json.loads(d_res.read().decode('utf-8'))
                
                deals = details_data.get('deals', [])
                lowest_price_usd = None
                lowest_deal_id = None
                lowest_store = None

                for deal in deals:
                    store_id = deal.get('storeID')
                    price_usd = float(deal.get('price', 999.0))
                    deal_id = deal.get('dealID')
                    
                    # Strict platform filtering
                    is_steam_deal = store_id in ["1", "2", "3", "11", "15", "18"]
                    is_epic_deal = store_id == "25"
                    is_xbox_deal = store_id == "27"
                    
                    if is_steam_deal and not item.check_steam: continue
                    if is_epic_deal and not item.check_epic: continue
                    if is_xbox_deal and not item.check_xbox: continue
                    
                    if not is_steam_deal and not is_epic_deal and not is_xbox_deal:
                        continue

                    if lowest_price_usd is None or price_usd < lowest_price_usd:
                        lowest_price_usd = price_usd
                        lowest_deal_id = deal_id
                        lowest_store = store_map.get(store_id, f"Store #{store_id}")

                item.last_checked_at = timezone.now()

                if lowest_price_usd is not None:
                    lowest_price_inr = lowest_price_usd * usd_to_inr
                    item.current_price = lowest_price_inr
                    item.lowest_platform = lowest_store
                    if lowest_deal_id:
                        item.buy_link = f"https://www.cheapshark.com/redirect?dealID={lowest_deal_id}"
                    else:
                        item.buy_link = None
                    
                    # Notify check in INR
                    if lowest_price_inr <= item.target_budget:
                        if not item.notified_under_budget:
                            subject = f"💸 DEAL ALERT: {item.name} is on sale!"
                            
                            # Envato-Style Premium Dark Theme HTML Email
                            buy_btn_html = f'<a href="{item.buy_link}" class="btn-cta" target="_blank" style="display: inline-block; background: linear-gradient(135deg, #a855f7, #6366f1); color: #ffffff !important; text-decoration: none !important; font-size: 14px; font-weight: 700; padding: 15px 40px; border-radius: 50px; text-transform: uppercase; letter-spacing: 1px; box-shadow: 0 0 15px rgba(168, 85, 247, 0.4); margin-top: 20px;">Purchase Deal</a>' if item.buy_link else ''
                            
                            game_thumb = matched_result.get('thumb', '')
                            thumb_img_html = f'<div style="text-align: center; margin: 20px 0;"><img src="{game_thumb}" alt="{item.name}" style="width: 140px; height: 190px; object-fit: cover; border-radius: 12px; border: 2px solid rgba(34, 211, 238, 0.4); box-shadow: 0 0 25px rgba(34, 211, 238, 0.25);"></div>' if game_thumb else ''

                            html_msg = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ margin: 0; padding: 0; background-color: #0c0a12; font-family: 'Inter', Helvetica, Arial, sans-serif; color: #e2e8f0; }}
        .wrapper {{ width: 100%; table-layout: fixed; background-color: #0c0a12; padding: 40px 0; }}
        .main-card {{ max-width: 600px; margin: 0 auto; background-color: #141020; border: 1px solid rgba(168, 85, 247, 0.2); border-radius: 16px; overflow: hidden; box-shadow: 0 10px 40px rgba(0,0,0,0.5); }}
        .header {{ background: linear-gradient(135deg, #a855f7, #6366f1); padding: 30px; text-align: center; }}
        .header h1 {{ margin: 0; font-size: 24px; font-weight: 800; letter-spacing: 2px; color: #ffffff; text-transform: uppercase; }}
        .content {{ padding: 40px 30px; text-align: center; }}
        .game-title {{ font-size: 28px; font-weight: 800; color: #22d3ee; margin: 10px 0 20px 0; text-shadow: 0 0 10px rgba(34, 211, 238, 0.3); }}
        .price-badge {{ display: inline-block; background: rgba(34, 197, 94, 0.15); border: 1px solid #22c55e; border-radius: 30px; padding: 10px 25px; font-size: 20px; font-weight: 700; color: #22c55e; margin-bottom: 25px; }}
        .details-table {{ width: 100%; border-collapse: collapse; margin: 25px 0; background: rgba(255,255,255,0.02); border-radius: 8px; overflow: hidden; }}
        .details-table td {{ padding: 12px 15px; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 14px; text-align: left; }}
        .details-table td.label {{ color: #94a3b8; font-weight: bold; width: 35%; }}
        .details-table td.value {{ color: #f1f5f9; }}
        .footer {{ background: #07050a; padding: 25px; text-align: center; font-size: 11px; color: #64748b; border-top: 1px solid rgba(255,255,255,0.03); }}
    </style>
</head>
<body>
    <div class="wrapper" style="background-color: #0c0a12; padding: 40px 0;">
        <div class="main-card" style="max-width: 600px; margin: 0 auto; background-color: #141020; border: 1px solid rgba(168, 85, 247, 0.2); border-radius: 16px; overflow: hidden;">
            <div class="header" style="background: linear-gradient(135deg, #a855f7, #6366f1); padding: 30px; text-align: center;">
                <h1 style="margin: 0; font-size: 24px; font-weight: 800; letter-spacing: 2px; color: #ffffff; text-transform: uppercase;">DEAL ALERT</h1>
            </div>
            <div class="content" style="padding: 40px 30px; text-align: center; color: #e2e8f0;">
                <p style="margin: 0; font-size: 12px; font-weight: 700; text-transform: uppercase; color: #a855f7; letter-spacing: 1px;">Target Budget Met</p>
                <div class="game-title" style="font-size: 28px; font-weight: 800; color: #22d3ee; margin: 10px 0 20px 0;">{item.name}</div>
                {thumb_img_html}
                <div class="price-badge" style="display: inline-block; background: rgba(34, 197, 94, 0.15); border: 1px solid #22c55e; border-radius: 30px; padding: 10px 25px; font-size: 20px; font-weight: 700; color: #22c55e; margin-bottom: 25px;">₹{lowest_price_inr:.2f}</div>
                <p style="font-size: 14px; color: #94a3b8; line-height: 1.6; margin: 0 0 20px 0;">Great news! A monitored title on your budget watchlist has dropped below your budget threshold.</p>
                
                <table class="details-table" style="width: 100%; border-collapse: collapse; margin: 25px 0; background: rgba(255,255,255,0.02); border-radius: 8px; overflow: hidden;">
                    <tr>
                        <td class="label" style="padding: 12px 15px; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 14px; text-align: left; color: #94a3b8; font-weight: bold; width: 35%;">Store</td>
                        <td class="value" style="padding: 12px 15px; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 14px; text-align: left; color: #f1f5f9;">{lowest_store}</td>
                    </tr>
                    <tr>
                        <td class="label" style="padding: 12px 15px; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 14px; text-align: left; color: #94a3b8; font-weight: bold; width: 35%;">Current Price</td>
                        <td class="value" style="padding: 12px 15px; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 14px; text-align: left; color: #f1f5f9;">₹{lowest_price_inr:.2f} (approx ${lowest_price_usd:.2f})</td>
                    </tr>
                    <tr>
                        <td class="label" style="padding: 12px 15px; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 14px; text-align: left; color: #94a3b8; font-weight: bold; width: 35%;">Target Budget</td>
                        <td class="value" style="padding: 12px 15px; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 14px; text-align: left; color: #f1f5f9;">₹{item.target_budget:.2f}</td>
                    </tr>
                </table>
                
                {buy_btn_html}
            </div>
            <div class="footer" style="background: #07050a; padding: 25px; text-align: center; font-size: 11px; color: #64748b; border-top: 1px solid rgba(255,255,255,0.03);">
                <p style="margin: 0 0 8px 0;">This is an automated notification from your QuantTrader Pro Arcade Lounge dashboard.</p>
                <p style="margin: 0;">© 2026 QuantTrader Pro. All Rights Reserved.</p>
            </div>
        </div>
    </div>
</body>
</html>
"""
                            
                            plain_body = f"Good news! {item.name} has dropped below your budget limit of INR {item.target_budget:.2f}:\n\nCurrent Price: INR {lowest_price_inr:.2f} (approx ${lowest_price_usd:.2f})\nStore: {lowest_store}\n\nGet it before the sale ends!"
                            self.send_email_notification(subject, plain_body, html_message=html_msg)
                            item.notified_under_budget = True
                    else:
                        item.notified_under_budget = False
                
                item.save()

            except Exception as e:
                self.stderr.write(f"Failed to check CheapShark sales for {item.name}: {e}")
