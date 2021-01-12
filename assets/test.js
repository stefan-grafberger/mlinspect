function hightightCodeBlock(editor, codeReferenceText){
    code_reference = JSON.parse(codeReferenceText);
    console.log(code_reference); // todo: call marktext
    document.querySelectorAll('.code-dag-highlight').forEach(block => {
        block.classList.remove("code-dag-highlight");
    });
    editor.markText(
        {line:(code_reference.lineno-1), ch:code_reference.col_offset},
        {line:(code_reference.end_lineno-1), ch:code_reference.end_col_offset},
        {className:  "code-dag-highlight"}
    );
}

setTimeout(function(){
    textArea = document.querySelector('#pipeline-textarea');
    console.log(textArea);
    // then highlight each
    var editor = CodeMirror.fromTextArea(textArea, {
        lineNumbers: true,
        matchBrackets: true,
        mode: "text/x-python",
        theme : "default"
        // can use other themes, only need to include css file from https://cdnjs.com/libraries/codemirror
        // see https://codemirror.net/demo/theme.html#default
    });

    editor.on('change', function(){
        // get value right from instance
        editor.save();
        textArea.value = editor.getValue();
        console.log(editor.getValue());
        input = document.querySelector('#pipeline-textarea');
        input.value = editor.getValue();
    });
    console.log(editor);

    //More Details https://developer.mozilla.org/en-US/docs/Web/API/MutationObserver
    // select the target node
    var hoverDiv = document.querySelector('#hovered-code-reference');
    var selectDiv = document.querySelector('#selected-code-reference');
    //console.log(target.getAttribute('n_clicks_timestamp')); // todo: get actual attributes
    // create an observer instance
    var observer = new MutationObserver(function(mutations) {
        console.log("attributes: ");
        if (hoverDiv.innerText) {
            hightightCodeBlock(editor, hoverDiv.innerText);
        } else if (selectDiv.innerText) {
            hightightCodeBlock(editor, selectDiv.innerText);
        } else {
            document.querySelectorAll('.code-dag-highlight').forEach(block => {
                block.classList.remove("code-dag-highlight");
            });
        }
    });
    // configuration of the observer:
    var config = { attributes: true, childList: true, characterData: true };
    // pass in the target node, as well as the observer options
    observer.observe(hoverDiv, config);
    observer.observe(selectDiv, config)
}, 2000);
