package main

import (		
		"bookstore/controllers"
		"bookstore/driver"		
		"log"
		"net/http"
		"github.com/gorilla/mux"				
		"github.com/subosito/gotenv"
)




func init() {
	gotenv.Load()
}

func logFatal(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func main() {
	
	router := mux.NewRouter()
	controller := controllers.Controller{}
	collection := driver.ConnectDB()
	
	router.HandleFunc("/books", controller.GetBooks(collection)).Methods("GET")
	router.HandleFunc("/books/{id}", controller.GetBook(collection)).Methods("GET")
	router.HandleFunc("/books", controller.AddBook(collection)).Methods("POST")
	router.HandleFunc("/books", controller.UpdateBook(collection)).Methods("PUT")
	router.HandleFunc("/books/{id}", controller.RemoveBook(collection)).Methods("DELETE")
	
	log.Println("REST end point Server is running at port 7000")
	log.Fatal(http.ListenAndServe(":7000", router))

}





