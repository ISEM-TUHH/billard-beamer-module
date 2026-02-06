function testAPI() {
	fetch("/api-doc")
		.then((r) => r.json())
		.then((a) => a.api)
		.then((api) => {
			console.log(api)
			di = document.getElementById("api-doc")
			for (let i=0; i < api.length; i++) {
				//console.log(api[i])
				/*var button = document.createElement("button");
				button.innerText = api[i];
				button.addEventListener("click", () => {
					fetch(api[i].substring(1))
						.then((r) => {
							try{
								return r.json()
							} catch (error) {
								console.log(error, r)
							}})
						.then((r) => {
							document.getElementById("api-out").innerText = api[i] + ": " + JSON.stringify(r,null, 2);
							console.log(r, api[i])
						})
				});
				di.appendChild(button)*/
				var a = document.createElement("a");
				a.innerText = api[i];
				a.href = api[i];
				a.target = "_blank";
				di.appendChild(a);
				di.appendChild(document.createElement("br"))
			}
		})
}

testAPI()


// manual config
document.getElementById("config").addEventListener("submit", function (e) {
	console.log("Hello world")
	e.preventDefault();
	
	form = document.getElementById("config")
	var formData = new FormData(form);
	// output as an object
	console.log(Object.fromEntries(formData));
	
	fetch("/v1/manin", {
		method: "POST",
		/*headers: {
			"Accept": "application/json",
			"Content-Type": "application/json"
		},*/
		body: formData
	})
		.then((res) => console.log(res))
		/*.then((res) => res.json())
		.then((e) => {
			console.log(e)
			document.getElementById("table-single").innerHTML = e["single-board"]
			document.getElementById("table-team").innerHTML = e["team-board"]

			updatePodium("single", e);
			updatePodium("team", e);

		})*/
});

document.getElementById("volume-slider").addEventListener("change", (e) => {
	var element = document.getElementById("volume-slider");
	console.log("Sound volume:", element.value);
	fetch("/v1/soundvolume",{
		method: "POST",
		headers: {
			"Accept": "content/plain",
			"Content-Type": "application/json"
		},
		body: JSON.stringify({"level": element.value})
	})
		.then((res) => {
			element.labels[0].innerText = element.value + "%";
		})
})