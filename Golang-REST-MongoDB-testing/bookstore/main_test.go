package main

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"bookstore/models"
	"encoding/json"
	"github.com/stretchr/testify/assert"
	
)
// ControllerMock
type ControllerMock struct {

   
}
// Our mocked ControllerMock method
func (m *ControllerMock) GetBooks() http.HandlerFunc {	

	return func (httpWriter http.ResponseWriter, httpRequest *http.Request) {
	// Create empty slice of struct pointers.
    books := []*models.Book{}

    // Create struct and append it to the slice.
    a1 := new(models.Book)
    a1.ID = 3
    a1.Title = "Book 1"
    a1.Name = "Book1 author"
	a1.Year = "2001"
    books = append(books, a1)

    // Create another struct.
    a2 := new(models.Book)
    a2.ID = 4
    a2.Title = "Book 4"
    a2.Name = "Book4 author"
	a2.Year = "2004"
    books = append(books, a2)
    json.NewEncoder(httpWriter).Encode(books)
	}
}
func (m *ControllerMock) DeleteBook(i int) http.HandlerFunc {	

	return func (httpWriter http.ResponseWriter, httpRequest *http.Request) {   

    json.NewEncoder(httpWriter).Encode(i)
	}
}
func (m *ControllerMock) GetBook(i int) http.HandlerFunc {	

	return func (httpWriter http.ResponseWriter, httpRequest *http.Request) {

    // Create struct and append it to the slice.	
	a1 := new(models.Book)

	if i==3 {
		a1.ID = 3
		a1.Title = "Book 1"
		a1.Name = "Book1 author"
		a1.Year = "2001"

	}

    json.NewEncoder(httpWriter).Encode(a1)
	}
}
func (m *ControllerMock) AddBook(book *models.Book) http.HandlerFunc {	

	return func (httpWriter http.ResponseWriter, httpRequest *http.Request) {
    json.NewEncoder(httpWriter).Encode(book.ID)
	}
}
func (m *ControllerMock) UpdateBook(book *models.Book) http.HandlerFunc {	

	return func (httpWriter http.ResponseWriter, httpRequest *http.Request) {
    json.NewEncoder(httpWriter).Encode(book.ID)
	}
}
func TestGetBooks(t *testing.T) {
	req, err := http.NewRequest("GET", "/books", nil)
	if err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	
	controller := new(ControllerMock)
	handler := http.HandlerFunc(controller.GetBooks())
	handler.ServeHTTP(rr, req)
	if status := rr.Code; status != http.StatusOK {
		t.Errorf("handler returned wrong status code: got %v want %v",
			status, http.StatusOK)
	}
	expected := "[{\"id\":3,\"title\":\"Book 1\",\"name\":\"Book1 author\",\"year\":\"2001\"},{\"id\":4,\"title\":\"Book 4\",\"name\":\"Book4 author\",\"year\":\"2004\"}]\n"
	assert.Equal(t, rr.Body.String(), expected)
	
}


func TestGetBooksByID(t *testing.T) {

	req, err := http.NewRequest("GET", "/books/3", nil)
	if err != nil {
		t.Fatal(err)
	}
	//req.URL.RawQuery = q.Encode()
	rr := httptest.NewRecorder()
	controller := new(ControllerMock)
	handler := http.HandlerFunc(controller.GetBook(3))
	handler.ServeHTTP(rr, req)
	if status := rr.Code; status != http.StatusOK {
		t.Errorf("handler returned wrong status code: got %v want %v",
			status, http.StatusOK)
	}
	expected := "{\"id\":3,\"title\":\"Book 1\",\"name\":\"Book1 author\",\"year\":\"2001\"}\n"
	assert.Equal(t, rr.Body.String(), expected)
}

func TestGetBooksByIDNotFound(t *testing.T) {

	req, err := http.NewRequest("GET", "/books/34", nil)
	if err != nil {
		t.Fatal(err)
	}	
	rr := httptest.NewRecorder()
	controller := new(ControllerMock)
	handler := http.HandlerFunc(controller.GetBook(34))
	handler.ServeHTTP(rr, req)
	if status := rr.Code; status != http.StatusOK {
		t.Errorf("handler returned wrong status code: got %v want %v",
			status, http.StatusOK)
	}
	expected := "{\"id\":0,\"title\":\"\",\"name\":\"\",\"year\":\"\"}\n"
	assert.Equal(t, rr.Body.String(), expected)
}

func TestAddBook(t *testing.T) {

	req, err := http.NewRequest("POST", "/books", nil)
	if err != nil {
		t.Fatal(err)
	}	
	a1 := new(models.Book)
	a1.ID = 3
	a1.Title = "Book 1"
	a1.Name = "Book1 author"
	a1.Year = "2001"
	rr := httptest.NewRecorder()
	controller := new(ControllerMock)
	handler := http.HandlerFunc(controller.AddBook(a1))
	handler.ServeHTTP(rr, req)
	if status := rr.Code; status != http.StatusOK {
		t.Errorf("handler returned wrong status code: got %v want %v",
			status, http.StatusOK)
	}
	expected := "3\n"
	assert.Equal(t, rr.Body.String(), expected)
}

func TestUpdateBook(t *testing.T) {

	req, err := http.NewRequest("PUT", "/books", nil)
	if err != nil {
		t.Fatal(err)
	}	
	a1 := new(models.Book)
	a1.ID = 3
	a1.Title = "Book 1"
	a1.Name = "Book3333 author"
	a1.Year = "2001"
	rr := httptest.NewRecorder()
	controller := new(ControllerMock)
	handler := http.HandlerFunc(controller.UpdateBook(a1))
	handler.ServeHTTP(rr, req)
	if status := rr.Code; status != http.StatusOK {
		t.Errorf("handler returned wrong status code: got %v want %v",
			status, http.StatusOK)
	}
	expected := "3\n"
	assert.Equal(t, rr.Body.String(), expected)
}
func TestRemoveBook(t *testing.T) {

	req, err := http.NewRequest("DELETE", "/books/3", nil)
	if err != nil {
		t.Fatal(err)
	}	
	
	rr := httptest.NewRecorder()
	controller := new(ControllerMock)
	handler := http.HandlerFunc(controller.DeleteBook(3))
	handler.ServeHTTP(rr, req)
	if status := rr.Code; status != http.StatusOK {
		t.Errorf("handler returned wrong status code: got %v want %v",
			status, http.StatusOK)
	}
	expected := "3\n"
	assert.Equal(t, rr.Body.String(), expected)
}

