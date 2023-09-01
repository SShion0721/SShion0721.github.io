document.addEventListener('DOMContentLoaded', () => {
    // Reveal animations on scroll
    const observerOptions = {
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, observerOptions);

    // Initial animations
    const revealElements = document.querySelectorAll('.reveal-text, .reveal-text-sub, .work-card');
    revealElements.forEach(el => observer.observe(el));

    // Parallax effect for "Louvre lines"
    window.addEventListener('scroll', () => {
        const scrolled = window.scrollY;
        const lines = document.querySelectorAll('.geometric-background div');
        lines.forEach((line, index) => {
            const speed = (index + 1) * 0.1;
            line.style.transform = `translate(-50%, -50%) rotate(45deg) translateY(${scrolled * speed}px)`;
        });
    });

    // Form submission handling
    const contactForm = document.getElementById('contact-form');
    if (contactForm) {
        contactForm.addEventListener('submit', (e) => {
            e.preventDefault();
            alert('感谢您的消息！这只是一个演示。');
            contactForm.reset();
        });
    }
});
