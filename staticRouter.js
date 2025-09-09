const express = require('express');
const router = express.Router();


router.get("/", (req, res) => {
    return res.render("home", { urls: [] }); // initially pass empty array
});

module.exports = router;