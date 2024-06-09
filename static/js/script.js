// Toggle mobile menu
const mobileMenuToggle = document.querySelector('.mobile-menu-toggle');
const mobileMenu = document.querySelector('.mobile-menu');

mobileMenuToggle.addEventListener('click', () => {
  mobileMenu.classList.toggle('active');
});

// Smooth scrolling for navigation links
const navLinks = document.querySelectorAll('nav a');

navLinks.forEach(link => {
  link.addEventListener('click', e => {
    e.preventDefault();
    const targetId = link.getAttribute('href');
    const targetElement = document.querySelector(targetId);
    targetElement.scrollIntoView({
      behavior: 'smooth'
    });
  });
});

// Animate on scroll
const observerOptions = {
  root: null,
  rootMargin: '0px',
  threshold: 0.2
};

const observerCallback = (entries, observer) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('animate');
      observer.unobserve(entry.target);
    }
  });
};

const observer = new IntersectionObserver(observerCallback, observerOptions);

const animateElements = document.querySelectorAll('.animate-on-scroll');
animateElements.forEach(element => {
  observer.observe(element);
});

// Parallax effect
window.addEventListener('scroll', () => {
  const parallaxElements = document.querySelectorAll('.parallax');
  parallaxElements.forEach(element => {
    const scrollPosition = window.pageYOffset;
    element.style.transform = `translateY(${scrollPosition * 0.5}px)`;
  });
});

// Form validation
const form = document.querySelector('form');
const formInputs = form.querySelectorAll('input, textarea');

form.addEventListener('submit', e => {
  let isFormValid = true;

  formInputs.forEach(input => {
    if (input.value.trim() === '') {
      isFormValid = false;
      input.classList.add('invalid');
    } else {
      input.classList.remove('invalid');
    }
  });

  if (!isFormValid) {
    e.preventDefault();
  }
});