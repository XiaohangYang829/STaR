function trigger(el, eventType) {
    if (typeof eventType === 'string' && typeof el[eventType] === 'function') {
        el[eventType]();
    } else {
        const event =
        eventType === 'string'
            ? new Event(eventType, {bubbles: true})
            : eventType;
        el.dispatchEvent(event);
    }
}

function fetchList() {
    if (window.location.href.includes(`Character`)) {
        return document.querySelectorAll(".product-results-holder .product-character")

    } else {
        return document.querySelectorAll(".product-results-holder .product-animation")

    }
}

const wait = (ms) => new Promise(resolve => setTimeout(resolve, ms));

const nextPage = async () => {
    var lastButton = document.querySelectorAll(".pagination.pagination-sm li:last-child a")
    trigger(lastButton[0], "click")
    await wait(5000)
}

const nextPageReturn = async () => {
    var lastButton = document.querySelectorAll(".pagination.pagination-sm li:last-child a")
    try {
        trigger(lastButton[0], "click")
    } catch (error) {
        HTMLFormControlsCollection.log('There is no next page!')
        return False
    }
    await wait(5000)
    return True
}

// add this to record motion categories
function saveTextToFile(text, fileName) {
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;

    // Trigger a click on the link to initiate the download
    a.click();

    // Clean up
    URL.revokeObjectURL(url);
}

const start = async () => {
    var list = fetchList();
    var count = 1
    var listOfCategory = []
    var errorCategory = []
    listOfCategory.push('NewPage')

    for (var i = 0; i <= list.length; i++) {
    if (i >= list.length) {
        await nextPage()
        count++
        list = fetchList()
        listOfCategory.push('NewPage')
        // stop the downloading when reach the page 26 (for 96 motion per page)
        if (count > 26) {
            // Convert the array to a JSON string
            // const jsonString = JSON.stringify(listOfCategory);
            // Store the JSON string in local storage
            // localStorage.setItem('listData', jsonString);
            // console.log('List of texts has been stored in local storage');
            saveTextToFile(listOfCategory, 'claire_motion_category.txt');
            saveTextToFile(errorCategory, 'claire_motion_category_error.txt');
            return alert("Done!")
        }
        if (list.length == 0) {
            return alert("Done!")
        }
        i = 0
        window.scrollTo(0, document.body.scrollHeight);
    }

    listOfCategory.push(list[i].innerText.replace(/\n/g, ' '))

    try {
        trigger(list[i], "click")
        await wait(5000)

        var inplace = document.querySelectorAll(`#site > div:nth-child(5) > div > div > div.product-preview-holder.col-sm-6 > div > div.editor.row.row-no-gutter > div.editor-sidebar.col-xs-4 > div.sidebar-list > div > div > div.animation-settings-list > div > label > input[name="inplace"]`)[0]
        if (inplace) {
            trigger(inplace, "click")
            await wait(800)
        }
    
        var download = document.querySelectorAll(".product-preview-holder .editor.row.row-no-gutter > div.editor-sidebar.col-xs-4 button")[0]
        if (download) {
            trigger(download, "click")
            await wait(1000)
        }

        // set skin to "without skin"
        (() => { var select = document.querySelectorAll("select.input-sm.form-control")[2]; select.value = false; select.dispatchEvent(new Event('change', { bubbles: true })); })();    
        // set frames per seconds to "60"
        (() => { var select = document.querySelectorAll("select.input-sm.form-control")[3]; select.value = 60; select.dispatchEvent(new Event('change', { bubbles: true })); })();    
        // set keyframe reduction to "0" (none)
        (() => { var select = document.querySelectorAll("select.input-sm.form-control")[4]; select.value = 0; select.dispatchEvent(new Event('change', { bubbles: true })); })();

        var download2 = document.querySelectorAll(".modal-footer .btn-primary")[0]
        trigger(download2, "click")
        await wait(8000)
    
        console.log(`Completed item ${i} of ${list.length - 1}`)
    } catch (error) {
        // Log the error to the console or any other logging mechanism
        console.error("Error:", error.message);

        // Continue with the next iteration of the loop
        errorCategory.push(list[i].innerText.replace(/\n/g, ' '))
    }
    }
}

start()
