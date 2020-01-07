let app = new Vue({
    el: '#root',
    data: {
        start: false,
        image: null,
        image_base64: '',
        result: '',
        predictions: [],
    },
    mounted() {
        eel.lm()();
    },
    methods: {
        onUpload(event) {
            const file = event.target.files[0];
            this.image = URL.createObjectURL(file);

            const reader = new FileReader();
            reader.onload = (e) => {                
                this.image_base64 = e.target.result
            }
            
            reader.readAsDataURL(file);
        },
        async predict() {
            console.log('H!');

            let klasses = await eel.predict(this.image_base64)();
            console.log(klasses);
            this.predictions = klasses;
        }
    }
})