const textarea = document.getElementById("post-input");
const previewDiv = document.getElementById("preview");
const statusDiv = document.getElementById("statusBadge");
const clasMessage = document.getElementById("classification_message");
const censoredDiv = document.getElementById("preview_FST");
const regexDiv = document.getElementById("preview_regex");


textarea.addEventListener("input", async () => {
    const content = textarea.value;

    try {
        const res = await fetch("/clasify", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ content })
        });

        const data = await res.json();

        if (data.status === "success") {
            previewDiv.innerHTML = data.preview;
            statusDiv.innerHTML = data.classification;
            clasMessage.innerHTML = data.classification_message;
            censoredDiv.innerHTML = data.censored;
            regexDiv.innerHTML = data.regex;





        } else {
            previewDiv.textContent = "...";
            statusDiv.textContent = "...";
            clasMessage.textContent = "...";
            censoredDiv.textContent="...";
            regexDiv.textContent = "...";


        }

    } catch(err) {

    }
});