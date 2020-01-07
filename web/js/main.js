let app = new Vue({
    el: '#root',
    data: {
        start: false,
        image: null,
        result: '',
        prediction: '',
    },
    mounted() {
        eel.lm()();
    },
    methods: {
        onUpload(event) {
            const file = event.target.files[0];
            this.image = URL.createObjectURL(file);
        },
        predict() {

        }


    }
})