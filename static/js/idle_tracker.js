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
        
        this.isLoggingOut = false;
        console.log("[SECURITY] IdleTelemetryEngine initializing (status=" + statusUrl + ")");

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

                // Fresh login bypass — server consumed the flag, clear client lock
                if (data.fresh_login) {
                    sessionStorage.removeItem('qt_locked');
                    this.isLocked = false;
                    const overlay = document.getElementById('global-lockscreen-overlay');
                    if (overlay) {
                        overlay.classList.remove('ls-visible', 'ls-hiding');
                    }
                }

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

    async checkGameSessionActive() {
        try {
            const response = await fetch('/relax/api/immersion-status/');
            const data = await response.json();
            return data.active === true;
        } catch (e) {
            return false;
        }
    }

    startEngine() {
        setInterval(() => {
            if (this.isLoggingOut) return;
            
            const idleTime = Date.now() - this.lastActivity;

            // Check if any threshold is crossed
            const crossedLock = !this.isLocked && this.lockThreshold > 0 && idleTime >= this.lockThreshold;
            const crossedLogout = this.logoutThreshold > 0 && idleTime >= this.logoutThreshold;

            if (crossedLock || crossedLogout) {
                this.checkGameSessionActive().then(isActive => {
                    if (isActive) {
                        // User is in a gaming session, reset inactivity timer
                        this.lastActivity = Date.now();
                        return;
                    }

                    if (crossedLogout) {
                        this.triggerLogout();
                        return;
                    }

                    if (crossedLock) {
                        this.triggerLock();
                    }
                });
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
            overlay.classList.remove('ls-hiding');
            overlay.classList.add('ls-visible');
            // Focus the input after the animation settles
            setTimeout(() => {
                const input = document.getElementById('pin-input');
                if (input) input.focus();
            }, 300);
        }
    }

    triggerLogout() {
        if (this.isLoggingOut) return;
        this.isLoggingOut = true;
        
        // Save current path to restore on re-login
        localStorage.setItem('qt_last_path', window.location.pathname + window.location.search);
        
        const form = document.createElement('form');
        form.method = 'POST';
        form.action = this.logoutUrl;
        
        const csrfInput = document.createElement('input');
        csrfInput.type = 'hidden';
        csrfInput.name = 'csrfmiddlewaretoken';
        
        // Handle case where cookie might not exist or be structured differently
        const cookieRow = document.cookie.split('; ').find(row => row.startsWith('csrftoken='));
        if (cookieRow) {
            csrfInput.value = cookieRow.split('=')[1];
        } else {
            // Fallback: look for csrf token in DOM
            const domCsrf = document.querySelector('[name=csrfmiddlewaretoken]');
            if (domCsrf) csrfInput.value = domCsrf.value;
        }
        
        form.appendChild(csrfInput);
        document.body.appendChild(form);
        form.submit();
    }
}
