// idle_tracker.js
// QuantTrader Pro Security Telemetry Engine

class IdleTelemetryEngine {
    constructor(statusUrl, logoutUrl) {
        this.statusUrl = statusUrl;
        this.logoutUrl = logoutUrl;
        
        this.lastActivity = Date.now();
        this.isLocked = sessionStorage.getItem('qt_locked') === 'true';
        this.hasPassword = false;
        
        // Defaults, will be overridden by fetch
        this.lockThreshold = 0;
        this.logoutThreshold = 0;

        this.initListeners();
        this.fetchConfig();
    }

    async fetchConfig() {
        try {
            const response = await fetch(this.statusUrl);
            const data = await response.json();
            if (data.status === 'success') {
                this.lockThreshold = data.idle_lock_minutes * 60 * 1000;
                this.logoutThreshold = data.idle_logout_minutes * 60 * 1000;
                this.hasPassword = data.has_password;

                // Update UI lock button state
                const lockBtn = document.getElementById('manual-lock-trigger');
                const lockIcon = document.getElementById('lock-icon-inner');
                if (lockBtn && lockIcon) {
                    if (!this.hasPassword) {
                        lockBtn.classList.add('opacity-50', 'cursor-not-allowed');
                        lockBtn.title = "Passcode not configured";
                        lockIcon.classList.remove('fa-lock');
                        lockIcon.classList.add('fa-unlock');
                    } else {
                        lockBtn.classList.remove('opacity-50', 'cursor-not-allowed');
                        lockBtn.title = "Lock Session";
                        lockIcon.classList.remove('fa-unlock');
                        lockIcon.classList.add('fa-lock');
                    }
                }

                this.startEngine();
                
                // If it was already locked in session storage, show overlay immediately
                if (this.isLocked) {
                    this.showLockscreenOverlay();
                }
            }
        } catch (e) {
            console.error("Failed to load security config", e);
        }
    }

    initListeners() {
        const events = ['mousemove', 'keydown', 'mousedown', 'touchstart', 'scroll'];
        const updateActivity = () => {
            if (!this.isLocked) {
                this.lastActivity = Date.now();
            }
        };

        events.forEach(e => document.addEventListener(e, updateActivity, { passive: true }));
    }

    startEngine() {
        setInterval(() => {
            const idleTime = Date.now() - this.lastActivity;

            // Phase 2: Complete Logout
            if (this.logoutThreshold > 0 && idleTime >= this.logoutThreshold) {
                this.triggerLogout();
                return;
            }

            // Phase 1: Lockscreen
            if (!this.isLocked && this.lockThreshold > 0 && idleTime >= this.lockThreshold) {
                this.triggerLock();
            }
            
        }, 5000);
    }

    triggerLock() {
        if (!this.hasPassword) return; // Don't lock if no password is set
        this.isLocked = true;
        sessionStorage.setItem('qt_locked', 'true');
        this.showLockscreenOverlay();
    }

    showLockscreenOverlay() {
        const overlay = document.getElementById('global-lockscreen-overlay');
        if (overlay) {
            overlay.classList.remove('hidden');
            overlay.classList.add('flex');
            setTimeout(() => {
                overlay.classList.remove('opacity-0', 'scale-105', 'scale-110');
                const input = document.getElementById('pin-input');
                if (input) input.focus();
            }, 10);
        }
    }

    triggerLogout() {
        const form = document.createElement('form');
        form.method = 'POST';
        form.action = this.logoutUrl;
        
        const csrfInput = document.createElement('input');
        csrfInput.type = 'hidden';
        csrfInput.name = 'csrfmiddlewaretoken';
        csrfInput.value = document.cookie.split('; ').find(row => row.startsWith('csrftoken=')).split('=')[1];
        
        form.appendChild(csrfInput);
        document.body.appendChild(form);
        form.submit();
    }
}
