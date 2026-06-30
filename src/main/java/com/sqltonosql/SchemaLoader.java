package com.sqltonosql;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import java.lang.reflect.Type;
import java.util.Map;

public class SchemaLoader {
    private static final Gson gson = new Gson();

    public static Schema loadSchema(String json) {
        Type mapType = new TypeToken<Map<String, Schema.CollectionInfo>>() {
        }.getType();
        Map<String, Schema.CollectionInfo> collections = gson.fromJson(json, mapType);

        Schema schema = new Schema();
        schema.setCollections(collections);
        return schema;
    }
}
