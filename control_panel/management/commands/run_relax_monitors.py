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

    def send_email_notification(self, subject, body):
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
                fail_silently=False
            )
            self.stdout.write(f"Dispatched email notification: {subject.encode('ascii', errors='replace').decode('ascii')}")
        except Exception as e:
            self.stderr.write(f"Failed to dispatch email: {str(e).encode('ascii', errors='replace').decode('ascii')}")

    def check_beta_recruitment(self):
        self.stdout.write("Scouting beta test signups...")
        # Target games like Genshin Impact and Wuthering Waves
        target_games = Game.objects.filter(name__icontains="genshin") | Game.objects.filter(name__icontains="wuthering") | Game.objects.filter(name__icontains="wuwa")
        
        for game in target_games:
            query = urllib.parse.quote(f"{game.name} beta test recruitment signup")
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
        
        # Store ID mapping of CheapShark
        # Store 1 = Steam, Store 25 = Epic Games, Store 27 = Xbox Store
        store_map = {
            "1": "Steam",
            "25": "Epic Games",
            "27": "Xbox Store",
            "18": "GOG"
        }

        for item in watchlist:
            # Search game by title on CheapShark
            encoded_title = urllib.parse.quote(item.name)
            search_url = f"https://www.cheapshark.com/api/1.0/games?title={encoded_title}"
            
            try:
                req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=8) as res:
                    search_data = json.loads(res.read().decode('utf-8'))
                
                if not search_data:
                    continue

                # Get game details by CheapShark Game ID
                game_id = search_data[0].get('gameID')
                details_url = f"https://www.cheapshark.com/api/1.0/games?id={game_id}"
                
                with urllib.request.urlopen(urllib.request.Request(details_url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=8) as d_res:
                    details_data = json.loads(d_res.read().decode('utf-8'))
                
                deals = details_data.get('deals', [])
                lowest_price_usd = None
                lowest_store = None

                for deal in deals:
                    store_id = deal.get('storeID')
                    price_usd = float(deal.get('price', 999.0))
                    
                    # Apply platform filter preferences
                    if store_id == "1" and not item.check_steam: continue
                    if store_id == "25" and not item.check_epic: continue
                    if store_id == "27" and not item.check_xbox: continue

                    if lowest_price_usd is None or price_usd < lowest_price_usd:
                        lowest_price_usd = price_usd
                        lowest_store = store_map.get(store_id, f"Store #{store_id}")

                item.last_checked_at = timezone.now()

                if lowest_price_usd is not None:
                    lowest_price_inr = lowest_price_usd * usd_to_inr
                    item.current_price = lowest_price_inr
                    item.lowest_platform = lowest_store
                    
                    # Notify check in INR
                    if lowest_price_inr <= item.target_budget:
                        if not item.notified_under_budget:
                            subject = f"💸 DEAL ALERT: {item.name} is on sale!"
                            body = f"Good news! {item.name} has dropped below your budget limit of INR {item.target_budget:.2f}:\n\nCurrent Price: INR {lowest_price_inr:.2f} (approx ${lowest_price_usd:.2f})\nStore: {lowest_store}\n\nGet it before the sale ends!"
                            self.send_email_notification(subject, body)
                            item.notified_under_budget = True
                    else:
                        # Reset notification flag if price goes back up
                        item.notified_under_budget = False
                
                item.save()

            except Exception as e:
                self.stderr.write(f"Failed to check CheapShark sales for {item.name}: {e}")
