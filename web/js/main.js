let app = new Vue({
    el: '#root',
    data: {
        start: false,
        loading: false,
        model: 'vector',
        directory: '/home/alex/code/fsr/src/corpus/others',
        query: '',
        configured: false,
        files: [],
        displayed_files: [],
        // result_size: 5,
        validated_dir: false,
        valid_dir: null,
    },
    computed: {
        dir_success: function() {
            return {
                'is-success': this.validated_dir && this.valid_dir,
                'is-error': this.validated_dir && !this.valid_dir
            }
        }
    },
    methods: {
        load_model(model) {
            this.model = model;
        },
        config() {
            // Is model loaded?
            eel.check_loaded()(loaded => {
                if (!loaded) {
                    this.load_model(this.model);
                }
                // Load model
                this.loading = true;
                eel.use_model(this.model)(() => {
                    this.loading = false;

                    if (!this.validated_dir) {
                        eel.validate_dir(this.directory)(is_valid => {
                            this.validated_dir = true;
                            this.valid_dir = is_valid;

                            if (this.valid_dir) {
                                // Change directory;
                                eel.change_directory(this.directory)();

                                // Allow queries
                                this.configured = true;
                            }
                        });
                    }
                    else {
                        // Change directory;
                        eel.change_directory(this.directory)();

                        // Allow queries
                        this.configured = true;
                    }
                });
            })
        },
        run_query() {
            this.loading = true;

            // Make the query if the retrieval model has been loaded
            eel.query(this.query)(files => {
                this.files = files;
                this.displayed_files = files.slice(0, this.relevant_size)
                this.loading = false;
                // console.log(files);
            });
        },
        load_files() {
            eel.extract_text(this.directory)(read_files);
        }
        // load_vector_model: function() {
        //   this.loading = true;
        //   this.model = 'vector';
        //   // Vector Model call here


        //   this.loading = false;
        // },
        // load_latent_model() {
        //   this.loading = true;
        //   this.model = 'latent semantic';


        //   // Latent Semantic model here
        //   this.loading = false;
        // }
    }
})


function read_files(files) {
    console.log(files);
}