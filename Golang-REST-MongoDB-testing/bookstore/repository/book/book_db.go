package bookMongoDB

import (
	"bookstore/models"
	"log"
	"context"	
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"go.mongodb.org/mongo-driver/bson"	
	
)

type BookRepository struct{}

func logFatal(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
//
//
// To Retrieve all the Books from the store
//
//
func (b BookRepository) GetBooks(collection *mongo.Collection) []*models.Book {
	findOptions := options.Find()
	//Setting the maximum number of records to be returned to 100
	findOptions.SetLimit(100)

	// Here's an array in which you can store the decoded documents
	var results []*models.Book

	// Passing bson.D{{}} as the filter matches all documents in the collection
	cur, err := collection.Find(context.TODO(), bson.D{{}}, findOptions)
	logFatal(err)

	// Finding multiple documents returns a cursor
	// Iterating through the cursor allows us to decode documents one at a time
	for cur.Next(context.TODO()) {		
		// create a value into which the single document can be decoded
		var book models.Book
		err := cur.Decode(&book)
		logFatal(err)
		results = append(results, &book)
	}
	err = cur.Err()
	logFatal(err)	
	// Close the cursor once finished
	cur.Close(context.TODO())
	log.Printf("Found multiple documents (array of pointers): %+v\n", results)
	log.Println("Get all books")	
	return results
}
func (b BookRepository) GetBook (collection *mongo.Collection, i int) models.Book{	
	filter := bson.M{"id": i}
	var book models.Book
	log.Println(filter)
	err := collection.FindOne(context.TODO(), filter).Decode(&book)
	logFatal(err)
	log.Printf("Found a single document: %+v\n", book)
	log.Println("Get book")	
	return book
}
func (b BookRepository) AddBook (collection *mongo.Collection, book models.Book) int{	
	insertResult, err1 := collection.InsertOne(context.TODO(), book)
	logFatal(err1)

	log.Println("Inserted a single document: ", insertResult.InsertedID)
	log.Println(book)
	log.Println("Add book")
	return book.ID
}
func (b BookRepository) UpdateBookName (collection *mongo.Collection, book models.Book) int{	
	filter := bson.M{"id": book.ID}
	//it wil update the name of the book in MongoDB
	update := bson.D{
    {"$set", bson.D{
        {"name", book.Name},
    }},
	}
	log.Println(filter)
	updateResult, err1 := collection.UpdateOne(context.TODO(), filter, update)
	logFatal(err1)

	log.Printf("Matched %v documents and updated %v documents.\n", updateResult.MatchedCount, updateResult.ModifiedCount)
	log.Println("update book")
	return book.ID
}
func (b BookRepository) RemoveBook (collection *mongo.Collection, i int) int{	
	filter := bson.M{"id": i}
	log.Println(filter)
	deleteResult, err := collection.DeleteOne(context.TODO(), filter)
	logFatal(err)
	log.Printf("Deleted %v documents in the books collection\n", deleteResult.DeletedCount)
	logFatal(err)
	log.Println("remove book")
	return i
}