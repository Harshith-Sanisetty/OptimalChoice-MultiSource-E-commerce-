const express = require("express");
const {handlegenerateshorturl,analytics} = require("../controllers/url");

const router = express.Router();

router.post("/",handlegenerateshorturl);

router.get("/analytics/:shortId",analytics);

module.exports = router;