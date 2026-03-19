window.HELP_IMPROVE_VIDEOJS = false;

// ========================================
// Copy BibTeX to clipboard
// ========================================
function copyBibTeX() {
  var bibtexElement = document.getElementById('bibtex-code');
  var button = document.querySelector('.copy-bibtex-btn');
  var copyText = button.querySelector('.copy-text');

  if (!bibtexElement) return;

  navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
    button.classList.add('copied');
    copyText.textContent = 'Copied!';
    setTimeout(function() {
      button.classList.remove('copied');
      copyText.textContent = 'Copy';
    }, 2000);
  }).catch(function() {
    var textArea = document.createElement('textarea');
    textArea.value = bibtexElement.textContent;
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand('copy');
    document.body.removeChild(textArea);
    button.classList.add('copied');
    copyText.textContent = 'Copied!';
    setTimeout(function() {
      button.classList.remove('copied');
      copyText.textContent = 'Copy';
    }, 2000);
  });
}

// ========================================
// Scroll to top
// ========================================
function scrollToTop() {
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

window.addEventListener('scroll', function() {
  var scrollButton = document.querySelector('.scroll-to-top');
  if (scrollButton) {
    if (window.pageYOffset > 300) {
      scrollButton.classList.add('visible');
    } else {
      scrollButton.classList.remove('visible');
    }
  }
});

// ========================================
// Method Cards: Pipeline Highlight
// ========================================
function setupMethodCards() {
  var cards = document.querySelectorAll('.method-card');
  cards.forEach(function(card) {
    var step = card.getAttribute('data-step');
    var highlight = document.getElementById('highlight-step' + step);

    card.addEventListener('mouseenter', function() {
      if (highlight) highlight.classList.add('active');
    });

    card.addEventListener('mouseleave', function() {
      if (highlight) highlight.classList.remove('active');
    });
  });
}

// ========================================
// Dataset Tabs
// ========================================
function switchDataset(dataset) {
  var tabs = document.querySelectorAll('.dataset-tab');
  tabs.forEach(function(tab) {
    tab.classList.toggle('active', tab.getAttribute('data-dataset') === dataset);
  });

  var contents = document.querySelectorAll('.dataset-content');
  contents.forEach(function(content) {
    content.style.display = 'none';
  });

  var target = document.getElementById('dataset-' + dataset);
  if (target) target.style.display = 'block';
}

// ========================================
// Video Segmentation Carousel
// ========================================
var currentVideoIndex = 0;
var videoSectionVisible = false;

function playActiveVideo() {
  if (!videoSectionVisible) return;
  var activeSlide = document.querySelector('.video-slide.active');
  if (!activeSlide) return;
  var video = activeSlide.querySelector('video');
  if (!video) return;

  // Reset to start and play
  video.currentTime = 0;
  video.muted = true;
  var playPromise = video.play();
  if (playPromise !== undefined) {
    playPromise.catch(function() {
      // Autoplay blocked — add a one-time click listener to start
      document.addEventListener('click', function startVideo() {
        video.play().catch(function() {});
        document.removeEventListener('click', startVideo);
      }, { once: true });
    });
  }
}

function pauseAllVideos() {
  var videos = document.querySelectorAll('.video-slide video');
  videos.forEach(function(v) {
    v.pause();
  });
}

function switchVideo(direction) {
  var slides = document.querySelectorAll('.video-slide');
  if (slides.length === 0) return;

  pauseAllVideos();

  currentVideoIndex = (currentVideoIndex + direction + slides.length) % slides.length;
  updateVideoSlide();
}

function goToVideo(index) {
  var slides = document.querySelectorAll('.video-slide');
  if (index < 0 || index >= slides.length) return;

  pauseAllVideos();
  currentVideoIndex = index;
  updateVideoSlide();
}

function updateVideoSlide() {
  var slides = document.querySelectorAll('.video-slide');
  var dots = document.querySelectorAll('.video-dot');

  slides.forEach(function(slide, i) {
    slide.classList.toggle('active', i === currentVideoIndex);
  });

  dots.forEach(function(dot, i) {
    dot.classList.toggle('active', i === currentVideoIndex);
  });

  playActiveVideo();
}

// ========================================
// Observe video section visibility
// ========================================
function setupVideoSectionObserver() {
  var videoSection = document.getElementById('video-results');
  if (!videoSection) return;

  var observer = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
      videoSectionVisible = entry.isIntersecting;
      if (entry.isIntersecting) {
        playActiveVideo();
      } else {
        pauseAllVideos();
      }
    });
  }, { threshold: 0.2 });

  observer.observe(videoSection);
}

// ========================================
// Initialize on DOM ready
// ========================================
document.addEventListener('DOMContentLoaded', function() {
  setupMethodCards();
  setupVideoSectionObserver();
});
