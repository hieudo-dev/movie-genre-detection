let app = new Vue({
    el: '#root',
    data: {
        image: null,
        result: '',
        prediction: '',
    },
    mounted() {
        eel.load_model()();
    },
    methods: {
        onUpload(event) {
            this.image = event.target.files[0]
            console.log(this.image)
        }
    }
})