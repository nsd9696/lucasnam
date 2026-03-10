// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-publications",
          title: "publications",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-tmi",
          title: "TMI",
          description: "Too much information about me.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/tmi/";
          },
        },{id: "post-cross-instance-kv-cache-sharing-for-disaggregated-llm-serving-cutting-ttft-with-mooncake-and-lmcache",
        
          title: "Cross-Instance KV Cache Sharing for Disaggregated LLM Serving: Cutting TTFT with Mooncake and...",
        
        description: "How cross-instance KV cache sharing with Mooncake + LMCache reduces TTFT by 24% in multi-instance disaggregated LLM serving",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/cross-instance-kv-cache-sharing-mooncake-lmcache/";
          
        },
      },{id: "post-nixl-for-kv-cache-in-disaggregated-serving",
        
          title: "NIXL for KV Cache in Disaggregated Serving",
        
        description: "How NIXL accelerates KV Cache transfer in Prefill/Decode disaggregated LLM serving, its architecture, vLLM integration, and a real-world memory leak debugging story",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/nixl-kv-cache-disaggregated-serving/";
          
        },
      },{id: "post-cuda-graph-in-vllm-eliminating-cpu-overhead-in-llm-inference",
        
          title: "CUDA Graph in vLLM: Eliminating CPU Overhead in LLM Inference",
        
        description: "How CUDA Graph reduces CPU launch overhead in LLM decode, memory management with Private Pools, and vLLM&#39;s graph capture modes",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/cuda-graph-in-vllm/";
          
        },
      },{id: "post-multi-node-p-d-disagg-vllm-serving-how-efa-works-compared-to-infiniband",
        
          title: "Multi-Node P/D Disagg vLLM Serving: How EFA Works Compared to InfiniBand?",
        
        description: "Multi-node GPU communication on AWS EFA, InfiniBand vs EFA comparison, and vLLM P/D Disagg setup",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/multi-node-gpu-networking-efa-srd-en/";
          
        },
      },{id: "post-moe-expert-ffn-backend-experts-implementation",
        
          title: "MoE Expert FFN Backend: experts_implementation",
        
        description: "Selecting Expert FFN computation backends (eager, batched_mm, grouped_mm) in HuggingFace Transformers and benchmarking with Solar-Open 100B",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/moe-experts-implementation-backend-en/";
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/the_godfather/";
            },},{id: "news-a-simple-inline-announcement",
          title: 'A simple inline announcement.',
          description: "",
          section: "News",},{id: "news-a-long-announcement-with-details",
          title: 'A long announcement with details',
          description: "",
          section: "News",handler: () => {
              window.location.href = "/news/announcement_2/";
            },},{id: "news-a-simple-inline-announcement-with-markdown-emoji-sparkles-smile",
          title: 'A simple inline announcement with Markdown emoji! :sparkles: :smile:',
          description: "",
          section: "News",},{id: "projects-project-1",
          title: 'project 1',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/1_project/";
            },},{id: "projects-project-2",
          title: 'project 2',
          description: "a project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/projects/2_project/";
            },},{id: "projects-project-3-with-very-long-name",
          title: 'project 3 with very long name',
          description: "a project that redirects to another website",
          section: "Projects",handler: () => {
              window.location.href = "/projects/3_project/";
            },},{id: "projects-project-4",
          title: 'project 4',
          description: "another without an image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/4_project/";
            },},{id: "projects-project-5",
          title: 'project 5',
          description: "a project with a background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/5_project/";
            },},{id: "projects-project-6",
          title: 'project 6',
          description: "a project with no image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/6_project/";
            },},{id: "projects-project-7",
          title: 'project 7',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/7_project/";
            },},{id: "projects-project-8",
          title: 'project 8',
          description: "an other project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/projects/8_project/";
            },},{id: "projects-project-9",
          title: 'project 9',
          description: "another project with an image 🎉",
          section: "Projects",handler: () => {
              window.location.href = "/projects/9_project/";
            },},{id: "teachings-data-science-fundamentals",
          title: 'Data Science Fundamentals',
          description: "This course covers the foundational aspects of data science, including data collection, cleaning, analysis, and visualization. Students will learn practical skills for working with real-world datasets.",
          section: "Teachings",handler: () => {
              window.location.href = "/teachings/data-science-fundamentals/";
            },},{id: "teachings-introduction-to-machine-learning",
          title: 'Introduction to Machine Learning',
          description: "This course provides an introduction to machine learning concepts, algorithms, and applications. Students will learn about supervised and unsupervised learning, model evaluation, and practical implementations.",
          section: "Teachings",handler: () => {
              window.location.href = "/teachings/introduction-to-machine-learning/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%79%6F%75@%65%78%61%6D%70%6C%65.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
