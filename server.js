const express = require("express");
const cors = require("cors");
const Sentiment = require("sentiment");
const natural = require("natural");
const path = require("path");

const app = express();

// Enable CORS for all origins
app.use(cors({
  origin: '*',
  methods: ['GET', 'POST'],
  exposedHeaders: ['Access-Control-Allow-Private-Network'],
}));

app.use(express.json());

const tokenizer = new natural.WordTokenizer();
const stemmer = natural.PorterStemmer;
const nounInflector = new natural.NounInflector();
const sentiment = new Sentiment();

// Load POS tagger
const base_folder = "node_modules/natural/lib/natural/brill_pos_tagger";
const rulesFilename = path.join(base_folder, "data/English/tr_from_posjs.txt");
const lexiconFilename = path.join(base_folder, "data/English/lexicon_from_posjs.json");
const defaultCategory = "N";

const lexicon = new natural.Lexicon(lexiconFilename, defaultCategory);
const rules = new natural.RuleSet(rulesFilename);
const tagger = new natural.BrillPOSTagger(lexicon, rules);

// POST API
app.post("/analyze", (req, res) => {
  const { text } = req.body;

  if (!text) {
    return res.status(400).json({ error: "Text is required" });
  }

  const result = sentiment.analyze(text);

  const sentimentMessage =
    result.score > 0
      ? "Happy Statement"
      : result.score < 0
      ? "Negative Statement"
      : "Neutral Statement";

  const tokens = tokenizer.tokenize(text);
  const stemmedWords = tokens.map((word) => stemmer.stem(word));
  const singularWords = tokens.map((word) => nounInflector.singularize(word));

  const taggedWords = tagger.tag(tokens).taggedWords;
  const nouns = taggedWords
    .filter((w) => w.tag.startsWith("NN")) // Nouns
    .map((w) => w.token);

  res.json({
    sentimentScore: result.score,
    sentimentMessage,
    tokens,
    stemmedWords,
    singularWords,
    nouns,
  });
});

// Simple homepage to check if server is running
app.get("/", (req, res) => {
  res.send("Server is up and running 🚀");
});

// Dynamic Port (important for Render)
const PORT = process.env.PORT || 5000;

app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
