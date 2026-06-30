package com.sqltonosql;

import org.springframework.web.bind.annotation.*;
import net.sf.jsqlparser.parser.CCJSqlParserUtil;
import net.sf.jsqlparser.statement.Statement;
import net.sf.jsqlparser.statement.select.Select;
import net.sf.jsqlparser.statement.insert.Insert;
import net.sf.jsqlparser.statement.update.Update;
import net.sf.jsqlparser.statement.delete.Delete;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import com.google.gson.Gson;
import com.google.gson.JsonObject;

@RestController
@RequestMapping("/api")
public class ConversionController {

    // URL of the Python Flask inference server (05_inference.py --serve)
    private static final String INFERENCE_API_URL = System.getenv("INFERENCE_API_URL") != null
            ? System.getenv("INFERENCE_API_URL")
            : "http://localhost:5000/api/malayalam-to-sql";

    private static final HttpClient httpClient = HttpClient.newBuilder()
            .version(HttpClient.Version.HTTP_1_1)
            .build();
    private static final Gson gson = new Gson();

    @PostMapping("/convert")
    public ConversionResponse convert(@RequestBody ConversionRequest request) {
        try {
            // 1. Load Schema (Optional usage for now, but good practice)
            Schema schema = SchemaLoader.loadSchema(request.getSchema());

            // 2. Parse SQL
            Statement statement = CCJSqlParserUtil.parse(request.getSql());
            MongoConverter converter = new MongoConverter();
            String mongoQuery;

            // 3. Dispatch based on Type
            if (statement instanceof Select) {
                QueryVisitor visitor = new QueryVisitor();
                ((Select) statement).accept(visitor);
                mongoQuery = converter.convert(visitor.getQueryModel());
            } else if (statement instanceof Insert) {
                mongoQuery = converter.convertInsert((Insert) statement);
            } else if (statement instanceof Update) {
                mongoQuery = converter.convertUpdate((Update) statement);
            } else if (statement instanceof Delete) {
                mongoQuery = converter.convertDelete((Delete) statement);
            } else {
                return new ConversionResponse(null,
                        "Error: Unsupported SQL statement type. Only SELECT, INSERT, UPDATE, DELETE are supported.");
            }

            return new ConversionResponse(mongoQuery, null);

        } catch (Throwable e) {
            e.printStackTrace(); // Print full error to console for debugging
            return new ConversionResponse(null, "Error during conversion: " + e.getMessage());
        }
    }

    /**
     * Full pipeline: Malayalam question → SQL (via LLM) → MongoDB NoSQL query.
     * Requires the Python inference server (05_inference.py --serve) to be running.
     */
    @PostMapping("/malayalam-to-nosql")
    public FullPipelineResponse malayalamToNosql(@RequestBody MalayalamRequest request) {
        try {
            // Step 1: Call the Python inference API to get SQL from Malayalam
            JsonObject payload = new JsonObject();
            payload.addProperty("question", request.getQuestion());
            payload.addProperty("schema", request.getSchema() != null ? request.getSchema() : "");

            HttpRequest httpRequest = HttpRequest.newBuilder()
                    .uri(URI.create(INFERENCE_API_URL))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(payload.toString()))
                    .build();

            HttpResponse<String> response = httpClient.send(httpRequest,
                    HttpResponse.BodyHandlers.ofString());

            if (response.statusCode() != 200) {
                return new FullPipelineResponse(null, null, null,
                        "LLM inference error (HTTP " + response.statusCode() + "): " + response.body());
            }

            JsonObject result = gson.fromJson(response.body(), JsonObject.class);

            if (result.has("error") && !result.get("error").isJsonNull()) {
                return new FullPipelineResponse(null, null, null,
                        "LLM error: " + result.get("error").getAsString());
            }

            String generatedSql = result.get("sql").getAsString();

            // Step 2: Convert SQL → MongoDB using existing pipeline
            Statement statement = CCJSqlParserUtil.parse(generatedSql);
            MongoConverter converter = new MongoConverter();
            String mongoQuery;

            if (statement instanceof Select) {
                QueryVisitor visitor = new QueryVisitor();
                ((Select) statement).accept(visitor);
                mongoQuery = converter.convert(visitor.getQueryModel());
            } else if (statement instanceof Insert) {
                mongoQuery = converter.convertInsert((Insert) statement);
            } else if (statement instanceof Update) {
                mongoQuery = converter.convertUpdate((Update) statement);
            } else if (statement instanceof Delete) {
                mongoQuery = converter.convertDelete((Delete) statement);
            } else {
                return new FullPipelineResponse(generatedSql, null, null,
                        "Unsupported SQL type generated by LLM");
            }

            return new FullPipelineResponse(generatedSql, mongoQuery, request.getQuestion(), null);

        } catch (Throwable e) {
            e.printStackTrace();
            String msg = e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName();
            // Common case: ConnectException when Python inference server is not running
            if (e instanceof java.net.ConnectException || msg.contains("Connection refused")) {
                msg = "Cannot connect to inference server at " + INFERENCE_API_URL +
                        ". Please ensure the Python Flask server (05_inference.py --serve) is running.";
            }
            return new FullPipelineResponse(null, null, null, "Pipeline error: " + msg);
        }
    }

    // ─── DTO Classes ────────────────────────────────────────

    // Existing: SQL → NoSQL request
    public static class ConversionRequest {
        private String sql;
        private String schema;

        public String getSql() {
            return sql;
        }

        public void setSql(String sql) {
            this.sql = sql;
        }

        public String getSchema() {
            return schema;
        }

        public void setSchema(String schema) {
            this.schema = schema;
        }
    }

    // Existing: SQL → NoSQL response
    public static class ConversionResponse {
        private String mongoQuery;
        private String error;

        public ConversionResponse(String mongoQuery, String error) {
            this.mongoQuery = mongoQuery;
            this.error = error;
        }

        public String getMongoQuery() {
            return mongoQuery;
        }

        public String getError() {
            return error;
        }
    }

    // New: Malayalam → NoSQL request
    public static class MalayalamRequest {
        private String question;
        private String schema;

        public String getQuestion() {
            return question;
        }

        public void setQuestion(String question) {
            this.question = question;
        }

        public String getSchema() {
            return schema;
        }

        public void setSchema(String schema) {
            this.schema = schema;
        }
    }

    // New: Full pipeline response (Malayalam → SQL → NoSQL)
    public static class FullPipelineResponse {
        private String generatedSql;
        private String mongoQuery;
        private String originalQuestion;
        private String error;

        public FullPipelineResponse(String generatedSql, String mongoQuery,
                String originalQuestion, String error) {
            this.generatedSql = generatedSql;
            this.mongoQuery = mongoQuery;
            this.originalQuestion = originalQuestion;
            this.error = error;
        }

        public String getGeneratedSql() {
            return generatedSql;
        }

        public String getMongoQuery() {
            return mongoQuery;
        }

        public String getOriginalQuestion() {
            return originalQuestion;
        }

        public String getError() {
            return error;
        }
    }
}
