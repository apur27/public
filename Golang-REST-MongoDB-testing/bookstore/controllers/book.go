package controllers
import (		
		"log"		
		"bookstore/models"	
		"bookstore/repository/book"
		"net/http"
		"go.mongodb.org/mongo-driver/mongo"
		"encoding/json"
		"github.com/gorilla/mux"
		"strconv"
)
type Controller struct {}



func logFatal(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
func (c Controller) GetBooks(collection  *mongo.Collection) http.HandlerFunc {

	return func (httpWriter http.ResponseWriter, httpRequest *http.Request) {
		var results []*models.Book
		bookRepo := bookMongoDB.BookRepository{}
		results = bookRepo.GetBooks(collection)		
		json.NewEncoder(httpWriter).Encode(results)
	}


}
func (c Controller) GetBook(collection  *mongo.Collection) http.HandlerFunc {

	return func (httpWriter http.ResponseWriter, httpRequest *http.Request) {
		// create a value into which the result can be decoded
		var book models.Book
		params := mux.Vars(httpRequest)
		log.Println(params)
		i, err := strconv.Atoi(params["id"])
		log.Println(i)
		logFatal(err)		
		bookRepo := bookMongoDB.BookRepository{}
		book = bookRepo.GetBook(collection, i)		
		json.NewEncoder(httpWriter).Encode(book)
	}


}


func (c Controller) AddBook(collection  *mongo.Collection) http.HandlerFunc {

	return func (httpWriter http.ResponseWriter, httpRequest *http.Request) {
		var book models.Book	
		json.NewDecoder(httpRequest.Body).Decode(&book)
		bookRepo := bookMongoDB.BookRepository{}
		bookID := bookRepo.AddBook(collection, book)		
		json.NewEncoder(httpWriter).Encode(bookID)
	}


}

func (c Controller) UpdateBook(collection  *mongo.Collection) http.HandlerFunc {

	return func (httpWriter http.ResponseWriter, httpRequest *http.Request) {		
		var book models.Book	
		json.NewDecoder(httpRequest.Body).Decode(&book)
		bookRepo := bookMongoDB.BookRepository{}
		bookID := bookRepo.UpdateBookName(collection, book)		
		json.NewEncoder(httpWriter).Encode(bookID)
	}


}

func (c Controller) RemoveBook(collection  *mongo.Collection) http.HandlerFunc {

	return func (httpWriter http.ResponseWriter, httpRequest *http.Request) {		
		params := mux.Vars(httpRequest)
		log.Println(params)
		i, err1 := strconv.Atoi(params["id"])
		log.Println(i)
		logFatal(err1)
		bookRepo := bookMongoDB.BookRepository{}
		bookID := bookRepo.RemoveBook(collection, i)		
		json.NewEncoder(httpWriter).Encode(bookID)
	}


}

