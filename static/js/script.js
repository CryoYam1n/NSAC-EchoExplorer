// Create dynamic star field
        function createStarField() {
            const starField = document.querySelector('.star-field');
            const numStars = 150;
            
            for (let i = 0; i < numStars; i++) {
                const star = document.createElement('div');
                star.className = 'star';
                star.style.left = Math.random() * 100 + '%';
                star.style.top = Math.random() * 100 + '%';
                star.style.width = star.style.height = Math.random() * 3 + 1 + 'px';
                star.style.animationDelay = Math.random() * 4 + 's';
                star.style.animationDuration = (Math.random() * 3 + 2) + 's';
                starField.appendChild(star);
            }
        }

        // Premium scroll animations with intersection observer
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -100px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                }
            });
        }, observerOptions);

        // Observe all premium sections
        document.querySelectorAll('.section-premium').forEach(el => {
            observer.observe(el);
        });

        // Advanced parallax effect for celestial objects
        let ticking = false;
        
        function updateAdvancedParallax() {
            const scrolled = window.pageYOffset;
            const premium = document.querySelectorAll('.celestial-premium, .celestial-orbit-premium');
            
            premium.forEach((element, index) => {
                const speed = (index + 1) * 0.03;
                const rotation = scrolled * 0.01;
                const yPos = -(scrolled * speed);
                element.style.transform = `translateY(${yPos}px) rotate(${rotation}deg)`;
            });
            
            ticking = false;
        }

        function requestAdvancedTick() {
            if (!ticking) {
                requestAnimationFrame(updateAdvancedParallax);
                ticking = true;
            }
        }

        // Premium cursor trail effect
        let cursorTrails = [];
        const maxTrails = 10;

        document.addEventListener('mousemove', (e) => {
            // Create new trail element
            const trail = document.createElement('div');
            trail.className = 'cursor-trail';
            trail.style.left = e.clientX - 10 + 'px';
            trail.style.top = e.clientY - 10 + 'px';
            document.body.appendChild(trail);
            
            cursorTrails.push(trail);
            
            // Remove old trails
            if (cursorTrails.length > maxTrails) {
                const oldTrail = cursorTrails.shift();
                oldTrail.remove();
            }
            
            // Fade out trail
            setTimeout(() => {
                trail.style.opacity = '0';
                setTimeout(() => {
                    if (trail.parentNode) {
                        trail.remove();
                    }
                }, 300);
            }, 100);
        });

        // Premium smooth scrolling with easing
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    const targetPosition = target.offsetTop - 100;
                    const startPosition = window.pageYOffset;
                    const distance = targetPosition - startPosition;
                    const duration = 1000;
                    let start = null;

                    function smoothScroll(timestamp) {
                        if (!start) start = timestamp;
                        const progress = timestamp - start;
                        const easeProgress = easeInOutCubic(progress / duration);
                        
                        window.scrollTo(0, startPosition + (distance * easeProgress));
                        
                        if (progress < duration) {
                            requestAnimationFrame(smoothScroll);
                        }
                    }

                    function easeInOutCubic(t) {
                        return t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1;
                    }

                    requestAnimationFrame(smoothScroll);
                }
            });
        });

        // Premium loading animations with staggered reveals
        window.addEventListener('load', () => {
            document.body.classList.add('loaded');
            createStarField();
            
            // Stagger card animations
            const cards = document.querySelectorAll('.card-premium');
            cards.forEach((card, index) => {
                setTimeout(() => {
                    card.style.animation = 'fadeIn 0.8s ease-out forwards';
                }, index * 150);
            });

            // Stagger section animations
            const sections = document.querySelectorAll('.section-premium');
            sections.forEach((section, index) => {
                setTimeout(() => {
                    section.classList.add('visible');
                }, index * 200);
            });
        });

        // Premium button interactions with advanced effects
        document.querySelectorAll('.btn-premium').forEach(btn => {
            btn.addEventListener('mouseenter', function() {
                this.style.transform = 'translateY(-3px) scale(1.02)';
                this.style.boxShadow = '0 20px 40px rgba(74, 34, 233, 0.6)';
            });

            btn.addEventListener('mouseleave', function() {
                this.style.transform = 'translateY(0) scale(1)';
                this.style.boxShadow = '0 15px 35px rgba(74, 34, 233, 0.5)';
            });
        });

        // Premium scroll event handler
        window.addEventListener('scroll', requestAdvancedTick);

        // Premium responsive design enhancements
        function handleResize() {
            const isMobile = window.innerWidth < 768;
            const premiumElements = document.querySelectorAll('.hover-lift-premium');
            
            if (isMobile) {
                premiumElements.forEach(el => {
                    el.style.transform = 'none';
                });
            }
        }

        window.addEventListener('resize', handleResize);
        handleResize(); // Call on load

        // Premium performance optimization
        const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
        if (prefersReducedMotion.matches) {
            document.documentElement.style.setProperty('--animation-duration', '0s');
        }

        console.log('🚀 Echo Explorer Advanced Premium Version Loaded Successfully!');
        console.log('🌟 Features: Advanced animations, premium interactions, stellar performance');