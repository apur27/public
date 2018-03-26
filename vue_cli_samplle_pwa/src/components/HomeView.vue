<template>
<div class="container">


<b-carousel id="carousel1"
                style="text-shadow: 1px 1px 2px #333;"
                controls
                indicators="false"
                background="#ababab"
                :interval="4000"
                img-width="300"
                img-height="300"
                v-model="slide"
                @sliding-start="onSlideStart"
                @sliding-end="onSlideEnd"
    >
<div v-for="picture in this.pictures" :key="picture.id">
      <b-carousel-slide>
        <img slot="img" class="d-block img-fluid w-100" width="500" height="200"
             :src="pictures[picture.id].url" alt="image slot">
      </b-carousel-slide>
  <!-- content
  <b-carousel-slide :caption="picture.info"
                        :text="picture.comment"
                        :img-src="picture.url"
      >
      </b-carousel-slide>
      -->
      <!-- content
  <img :src="picture.url" title="picture.info">
  -->
</div>
</b-carousel>





</div>
</template>
<script>
  import data from '../data'
  export default {
    methods: {
      displayDetails (id) {
        this.$router.push({name: 'Detail', params: { id: id }})
      },
      onSlideStart (slide) {
        this.sliding = true
      },
      onSlideEnd (slide) {
        this.sliding = false
      }
    },
    data () {
      return {
        slide: 0,
        sliding: null,
        'pictures': data.pictures
      }
    }
  }
</script>
<style scoped>
  .add-picture-button {
    position: fixed;
    right: 24px;
    bottom: 24px;
    z-index: 998;
  }
  .image-card {
    position: relative;
    margin-bottom: 8px;
  }
  .image-card__picture > img {
    width:100%;
  }
  .image-card__comment {
    position: absolute;
    bottom: 0;
    height: 52px;
    padding: 16px;
    text-align: right;
    background: rgba(0, 0, 0, 0.5);
  }
  .image-card__comment > span {
    color: #fff;
    font-size: 14px;
    font-weight: bold;
  }
</style>
