setTimeout(function(){
   document.querySelectorAll('#pipeline-textarea').forEach(block => {

        // then highlight each
        var editor = CodeMirror.fromTextArea(block, {
            lineNumbers: true,
            matchBrackets: true,
            mode: "text/x-python",
            theme : "default"
            // can use other themes, only need to include css file from https://cdnjs.com/libraries/codemirror
            // see https://codemirror.net/demo/theme.html#default
        });

        document.addEventListener("keydown", keyDownTextField, false);
        function keyDownTextField(e) {
        var keyCode = e.keyCode;
          if(keyCode==13) { // Enter key
            editor.markText({line:1,ch:0},{line:10,ch:0},{className:  "code-dag-highlight"});
          }
        }

        //More Details https://developer.mozilla.org/en-US/docs/Web/API/MutationObserver
        // select the target node
        var target = document.querySelector('#pipeline-textarea') // todo: use hidden div
        console.log("attributes: ");
        console.log(target.attributes);
        console.log(target.getAttribute('n_clicks_timestamp')); // todo: get actual attributes
        // create an observer instance
        var observer = new MutationObserver(function(mutations) {
          console.log(target.innerText); // todo: call marktext
        });
        // configuration of the observer:
        var config = { attributes: true, childList: true, characterData: true };
        // pass in the target node, as well as the observer options
        observer.observe(target, config);
    });

}, 2000);