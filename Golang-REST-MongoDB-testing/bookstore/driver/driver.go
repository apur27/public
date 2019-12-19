package driver

import (		
		"os"
		"log"
		"context"
		"go.mongodb.org/mongo-driver/mongo"
		"go.mongodb.org/mongo-driver/mongo/options"
)
func logFatal(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func ConnectDB() *mongo.Collection   {
	mongoUrl := os.Getenv("MONGODB_URL")
	log.Println(mongoUrl)
	log.Printf("MongoDB URL: %+v\n", mongoUrl)
	// Set client options
	clientOptions := options.Client().ApplyURI(mongoUrl)

	// Connect to MongoDB
	client, err := mongo.Connect(context.TODO(), clientOptions)
	logFatal(err)

	// Check the connection
	err = client.Ping(context.TODO(), nil)
	logFatal(err)
	log.Println("Connected to MongoDB!")
	
	collection := client.Database("bookstore").Collection("book")
	return collection;
	
}