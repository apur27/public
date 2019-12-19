package utils

import (
		"encoding/json";
		"bookstore/driver"
		"bookstore/models"
		"net/http";
)

func SendError(httpWriter http.ResponseWriter, status int, err models.Error) {
	httpWriter.WriteHeader(status)
	json.NewEncoder(httpWriter).Encode(err)
}

func SendSuccess(httpWriter http.ResponseWriter, data interface{}) {
	json.NewEncoder(httpWriter).Encode(data)
}
