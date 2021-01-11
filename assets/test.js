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
    });

}, 2000);