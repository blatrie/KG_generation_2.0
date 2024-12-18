(function() {
  "use strict";

  /**
   * Easy selector helper function
   */
  const select = (el, all = false) => {
    el = el.trim()
    if (all) {
      return [...document.querySelectorAll(el)]
    } else {
      return document.querySelector(el)
    }
  }

  /**
   * Easy event listener function
   */
  const on = (type, el, listener, all = false) => {
    let selectEl = select(el, all)
    if (selectEl) {
      if (all) {
        selectEl.forEach(e => e.addEventListener(type, listener))
      } else {
        selectEl.addEventListener(type, listener)
      }
    }
  }

  /**
   * Easy on scroll event listener 
   */
  const onscroll = (el, listener) => {
    el.addEventListener('scroll', listener)
  }

  /**
   * Navbar links active state on scroll
   */
  let navbarlinks = select('#navbar .scrollto', true)
  const navbarlinksActive = () => {
    let position = window.scrollY + 200
    navbarlinks.forEach(navbarlink => {
      if (!navbarlink.hash) return
      let section = select(navbarlink.hash)
      if (!section) return
      if (position >= section.offsetTop && position <= (section.offsetTop + section.offsetHeight)) {
        navbarlink.classList.add('active')
      } else {
        navbarlink.classList.remove('active')
      }
    })
  }
  window.addEventListener('load', navbarlinksActive)
  onscroll(document, navbarlinksActive)

  /**
   * Scrolls to an element with header offset
   */
  const scrollto = (el) => {
    let elementPos = select(el).offsetTop
    window.scrollTo({
      top: elementPos,
      behavior: 'smooth'
    })
  }

  /**
   * Back to top button
   */
  let backtotop = select('.back-to-top')
  if (backtotop) {
    const toggleBacktotop = () => {
      if (window.scrollY > 100) {
        backtotop.classList.add('active')
      } else {
        backtotop.classList.remove('active')
      }
    }
    window.addEventListener('load', toggleBacktotop)
    onscroll(document, toggleBacktotop)
  }

  /**
   * Mobile nav toggle
   */
  on('click', '.mobile-nav-toggle', function(e) {
    select('body').classList.toggle('mobile-nav-active')
    this.classList.toggle('bi-list')
    this.classList.toggle('bi-x')
  })

  /**
   * Scrool with ofset on links with a class name .scrollto
   */
  on('click', '.scrollto', function(e) {
    if (select(this.hash)) {
      e.preventDefault()

      let body = select('body')
      if (body.classList.contains('mobile-nav-active')) {
        body.classList.remove('mobile-nav-active')
        let navbarToggle = select('.mobile-nav-toggle')
        navbarToggle.classList.toggle('bi-list')
        navbarToggle.classList.toggle('bi-x')
      }
      scrollto(this.hash)
    }
  }, true)

  /**
   * Scroll with ofset on page load with hash links in the url
   */
  window.addEventListener('load', () => {
    if (window.location.hash) {
      if (select(window.location.hash)) {
        scrollto(window.location.hash)
      }
    }
  });

  /**
   * Preloader
   */
  let preloader = select('#preloader');
  if (preloader) {
    window.addEventListener('load', () => {
      preloader.remove()
    });
  }

  /**
   * Hero type effect
   */
  const typed = select('.typed')
  if (typed) {
    let typed_strings = typed.getAttribute('data-typed-items')
    typed_strings = typed_strings.split(',')
    new Typed('.typed', {
      strings: typed_strings,
      loop: true,
      typeSpeed: 100,
      backSpeed: 50,
      backDelay: 2000
    });
  }

  /**
   * Skills animation
   */
  let skilsContent = select('.skills-content');
  if (skilsContent) {
    new Waypoint({
      element: skilsContent,
      offset: '80%',
      handler: function(direction) {
        let progress = select('.progress .progress-bar', true);
        progress.forEach((el) => {
          el.style.width = el.getAttribute('aria-valuenow') + '%'
        });
      }
    })
  }

  /**
   * Porfolio isotope and filter
   */
  window.addEventListener('load', () => {
    let portfolioContainer = select('.portfolio-container');
    if (portfolioContainer) {
      let portfolioIsotope = new Isotope(portfolioContainer, {
        itemSelector: '.portfolio-item'
      });

      let portfolioFilters = select('#portfolio-flters li', true);

      on('click', '#portfolio-flters li', function(e) {
        e.preventDefault();
        portfolioFilters.forEach(function(el) {
          el.classList.remove('filter-active');
        });
        this.classList.add('filter-active');

        portfolioIsotope.arrange({
          filter: this.getAttribute('data-filter')
        });
        portfolioIsotope.on('arrangeComplete', function() {
          AOS.refresh()
        });
      }, true);
    }

  });

  /**
   * Initiate portfolio lightbox 
   */
  const portfolioLightbox = GLightbox({
    selector: '.portfolio-lightbox'
  });

  /**
   * Initiate portfolio details lightbox 
   */
  const portfolioDetailsLightbox = GLightbox({
    selector: '.portfolio-details-lightbox',
    width: '90%',
    height: '90vh'
  });

  /**
   * Portfolio details slider
   */
  new Swiper('.portfolio-details-slider', {
    speed: 400,
    loop: true,
    autoplay: {
      delay: 5000,
      disableOnInteraction: false
    },
    pagination: {
      el: '.swiper-pagination',
      type: 'bullets',
      clickable: true
    }
  });

  /**
   * Testimonials slider
   */
  new Swiper('.testimonials-slider', {
    speed: 600,
    loop: true,
    autoplay: {
      delay: 5000,
      disableOnInteraction: false
    },
    slidesPerView: 'auto',
    pagination: {
      el: '.swiper-pagination',
      type: 'bullets',
      clickable: true
    }
  });

  /**
   * Animation on scroll
   */
  window.addEventListener('load', () => {
    AOS.init({
      duration: 1000,
      easing: 'ease-in-out',
      once: true,
      mirror: false
    })
  });

  /**
   * Initiate Pure Counter 
   */
  new PureCounter();



  document.getElementById("search-button").addEventListener('click', () => {

    // automaticaly scroll down to iframe 'id=htmlFrame' in the page
    const iframe = document.getElementById("htmlFrame");
    // TODO : code showGraph
    showGraph();
    // change iframe to be visible
    iframe.style.display = "block";
    // scrolling="auto" width="100%" height="500"
    iframe.setAttribute("scrolling", "auto");
    iframe.setAttribute("width", "100%");
    iframe.setAttribute("height", "750px");
    // scroll to iframe
    iframe.scrollIntoView();



    const htmlFrame = document.getElementById("htmlFrame");
    function loadAndRenderHTML() {
        // Fetch the HTML content from Flask

        fetch('/generate_html', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({search_term: ""}),
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.text(); // Assuming the server returns HTML content
            })
            .then(htmlContent => {
                const iframeDocument = htmlFrame.contentDocument || htmlFrame.contentWindow.document;
                iframeDocument.open();
                iframeDocument.write(htmlContent);
                iframeDocument.close();
            })
            .catch(error => {
                console.error(error);
            });

    }
    if (document.querySelector('#searchInput').value == "" && document.querySelector('#radius').value == "") {
      console.log("No search term or radius entered");
      loadAndRenderHTML()
    }
    else {
      console.log("Search term or radius entered");
    }
  });

  





  // SEARCH BAR CODE


  // FETCH ENTITIES HERE
  async function fetchEntities() {
    try {
        const response = await fetch('/get_entities', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const entities = await response.json();
        return entities;
    } catch (error) {
        console.error(error);
        throw error;  // Re-throw the error to be caught by the caller
    }
  }


  var database = []
  // Populate Database
  async function initializeDatabase() {
      try {
          let response = await fetchEntities();
          response.forEach(entity => {
              database.push({ word: entity[0], type: entity[1] });
          })
          return response;
      } catch (error) {
          // Handle error if needed
          console.error(error);
      }
  }

  // FORMAT :  { word: 'example1', type: 'example_type' },
  initializeDatabase();

  const suggestedButtonsDiv = document.getElementById('suggestedButtons');

  // Function to filter suggestions based on user input
  function filterSuggestions(input) {
    // FILTER WORDS HERE
    const filteredWords = database.filter(entry => entry.word.toLowerCase().includes(input.toLowerCase()));
    return filteredWords;
  }

  function switchPLaceholderGlow(valueStr) {
    let placehoderGlowDiv = document.querySelector('.placeholder-glow')
    suggestedButtonsDiv.innerHTML = '';

    if (valueStr === "hidden") {
      placehoderGlowDiv.classList.remove('shown');
      placehoderGlowDiv.classList.add('hidden');
    } else if (valueStr === "shown") {
      placehoderGlowDiv.classList.remove('hidden');
      placehoderGlowDiv.classList.add('shown');
    }
  }

  function displaySuggestions() {
    const input = document.querySelector('#searchInput').value;
  
    if (input.length !== 0) {
      const suggestions = filterSuggestions(input);

      // if suggestion is empty then keep placeholderGlow "shown" and return nothing
      if (suggestions.length === 0) {
        switchPLaceholderGlow("shown");
        return;
      } else {
        switchPLaceholderGlow("hidden");
        suggestions.forEach((entry, index) => {
            const highlightedWord = entry.word.replace(new RegExp(input, 'gi'), match => `${match}`);
        
            const button = createSuggestionButton(highlightedWord, entry.type);
            suggestedButtonsDiv.appendChild(button);
            suggestedButtonsDiv.style.overflowY = 'scroll';
        
            // Triggering reflow to apply transition on dynamically added elements
            void button.offsetWidth;
        
            // Set a timeout to add a class after 300ms
            setTimeout(() => {
              button.classList.add('visible');
            }, 300);
        
            button.addEventListener('click', () => addSelectedWord(entry.word));
          });
      }

      
    } else {
      switchPLaceholderGlow("hidden");
      // GET ALL ENTITIES
      setTimeout(() => {
        database.forEach(entity => {
          const button = createSuggestionButton(entity.word, entity.type);
          suggestedButtonsDiv.appendChild(button);
          suggestedButtonsDiv.style.overflowY = 'scroll';
          suggestedButtonsDiv.style.overflowX = 'hidden';
      
          // Triggering reflow to apply transition on dynamically added elements
          void button.offsetWidth;
      
          // Set a timeout to add a class after 300ms
          setTimeout(() => {
            button.classList.add('visible');
          }, 300);
      
          button.addEventListener('click', () => addSelectedWord(entity.word));
        });
      }, 200)
      
    }
    
  }
  
  function createSuggestionButton(highlightedWord, type) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'btn btn-light mb-2 p-2 col-md-12 text-left d-flex shadow bg-white rounded suggestion';

    const wordElement = document.createElement('span');
    wordElement.className = 'font-weight-bold col-md-8 text-justify';
    wordElement.innerText = highlightedWord;

    const typeElement = document.createElement('span');
    typeElement.className = 'font-weight-light font-italic col-md-2 text-right bg-light rounded p-1';
    typeElement.style.opacity = '0.5';
    typeElement.innerText = type;

    // Add some space between the two elements
    const spaceElement = document.createTextNode(' ');

    // Append the elements to the button
    button.appendChild(wordElement);
    button.appendChild(spaceElement);
    button.appendChild(typeElement);

    return button;
  }
  

  function addSelectedWord(word) {
    const searchInput = document.getElementById('searchInput');

    // Append the highlighted word to the search input
    searchInput.value = word;

    // Clear the suggestions and reset the input value for better user experience
    suggestedButtonsDiv.innerHTML = '';
  }

  // Function to handle the search and show graph
  function showGraph() {
    const searchInput = document.querySelector('#searchInput').value;
    const groupBy = document.querySelector('#groupBy').value;
    const radius = document.querySelector('#radius').value;

    // Implement your logic to show the graph based on the input parameters
    console.log(`Search Input: ${searchInput}, Group By: ${groupBy}, Radius: ${radius}`);
    // post request to flask
    
    fetch('/load_data_partially', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json',
      },
      body: JSON.stringify({search_term: searchInput, group_by: groupBy, radius: radius}),
    })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.text(); // Assuming the server returns HTML content
        })
        .then(htmlContent => {
            const iframeDocument = htmlFrame.contentDocument || htmlFrame.contentWindow.document;
            iframeDocument.open();
            iframeDocument.write(htmlContent);
            iframeDocument.close();
        })
        .catch(error => {
            console.error(error);
        });
  }

  // Attach event listeners
  document.querySelector('#searchInput').addEventListener('input', displaySuggestions);
  switchPLaceholderGlow("shown");
  // GET ALL ENTITIES
  setTimeout(() => {
    database.forEach(entity => {
      const button = createSuggestionButton(entity.word, entity.type);
      suggestedButtonsDiv.appendChild(button);
      suggestedButtonsDiv.style.overflowY = 'scroll';
      suggestedButtonsDiv.style.overflowX = 'hidden';
  
      // Triggering reflow to apply transition on dynamically added elements
      void button.offsetWidth;
  
      // Set a timeout to add a class after 300ms
      setTimeout(() => {
        button.classList.add('visible');
      }, 300);
  
      button.addEventListener('click', () => addSelectedWord(entity.word));
    });
  }, 200)
  switchPLaceholderGlow("hidden");
})()


