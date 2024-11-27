// Fade-in effect for the result section
document.addEventListener('DOMContentLoaded', function () {
    const resultCard = document.querySelector('.result-card');
    if (resultCard) {
        resultCard.classList.add('fade-in');
    }
});

// CSS for fade-in animation
document.styleSheets[0].insertRule(`
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
`, document.styleSheets[0].cssRules.length);
