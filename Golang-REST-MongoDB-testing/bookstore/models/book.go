package models

type Book struct {
	ID int `json:"id"`
	Title string `json:"title"`
	Name string `json:"name"`
	Year string `json:"year"`
}
