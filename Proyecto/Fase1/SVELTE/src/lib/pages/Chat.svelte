<script>
	import { tick } from "svelte";
    import { v4 as uuid } from "uuid";
    let input = $state();
    let inputText = $state('');
    let isLoading = $state(false);
    let disableAdding = $state(false);
    let isAdding = $state(false);
    let scroll = $state();
    let autoscroll = $state();
    let listDiv = $state()
    let listDivScrollHeight = $state()

    function clearInput() {
        inputText = '';
    }

    function focusInput() {
        input.focus();
    }

    let todos = $state([
        { id: uuid(), user: 'C', message: 'Lorem ipsun asfasdf' },
        { id: uuid(), user: 'U', message: 'ASDF fojsl ' }
    ])

    $effect(() => {
        todos && todos.length;
            if(autoscroll) listDiv.scrollTo(0, listDivScrollHeight);
            autoscroll = false;
    })

    async function responseQuestion() {
        isAdding = true;
        todos = [...todos, { id: uuid(), user: 'C', message: 'OK' }];
        isAdding = false;
        await tick();
        focusInput();
    }

    async function handleAddQuestion(event) {
        event.preventDefault();
        isAdding = true;
        todos = [...todos, { id: uuid(), user: 'U', message: inputText}]
        clearInput();
        isAdding = false
        await tick();
        setTimeout(() => {
            responseQuestion();
        }, 1000)
        focusInput();
    }

    
</script>


<!-- Chat -->
<div class="flex flex-col flex-auto w-500 h-full p-1">
	<div class="flex flex-col flex-auto flex-shrink-0 rounded-2xl bg-gray-100 h-full p-4">
		<div 
            class="flex flex-col h-full overflow-x-auto mb-4"
            bind:this={listDiv}
        >
			<div
                class="flex flex-col h-full"
                bind:this={listDivScrollHeight}
            >
				<div class="grid grid-cols-12 gap-y-2">
                    {#each todos as todo, index (todo.id)}
                    {@const { id, user, message } = todo}
                        {#if user === 'C'}
                            <div class="col-start-1 col-end-8 p-3 rounded-lg">
                                <div class="flex flex-row items-center">
                                    <div
                                        class="flex items-center justify-center h-10 w-10 rounded-full bg-indigo-500 flex-shrink-0"
                                    >
                                        {user}
                                    </div>
                                    <div class="relative ml-3 text-sm bg-white py-2 px-4 shadow rounded-xl">
                                        <div>
                                            {message}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        {:else if user === 'U'}
                            <div class="col-start-6 col-end-13 p-3 rounded-lg">
                                <div class="flex items-center justify-start flex-row-reverse">
                                    <div
                                        class="flex items-center justify-center h-10 w-10 rounded-full bg-indigo-500 flex-shrink-0"
                                    >
                                        {user}
                                    </div>
                                    <div class="relative mr-3 text-sm bg-indigo-100 py-2 px-4 shadow rounded-xl">
                                        <div>{message}</div>
                                        <div class="absolute text-xs bottom-0 right-0 -mb-5 mr-2 text-gray-500">Seen</div>
                                    </div>
                                </div>
                            </div>
                        {/if}
                    {/each}
				</div>
			</div>
		</div>
		<form
            class="flex flex-row items-center h-16 rounded-xl bg-gray-400 w-full px-4"
            onsubmit={handleAddQuestion}
        >
			<div class="flex-grow ml-4">
				<div class="relative w-full">
					<input
                        disabled={disableAdding || !todos}
						type="text"
						class="flex w-full border rounded-xl focus:outline-none focus:border-indigo-300 pl-4 h-10"
                        bind:this={input}
                        bind:value={inputText}
                        placeholder="Ask a Question"
					/>
				</div>
			</div>
			<div class="ml-4">
				<button
					class="flex items-center justify-center bg-indigo-500 hover:bg-indigo-600 rounded-xl text-white px-4 py-1 flex-shrink-0"
                    type="submit"
                    disabled={!inputText || disableAdding || !todos} 
				>
					<span>Send</span>
					<span class="ml-2">
						<svg
							class="w-4 h-4 transform rotate-45 -mt-px"
							fill="none"
							stroke="currentColor"
							viewBox="0 0 24 24"
							xmlns="http://www.w3.org/2000/svg"
						>
							<path
								stroke-linecap="round"
								stroke-linejoin="round"
								stroke-width="2"
								d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
							/>
						</svg>
					</span>
				</button>
			</div>
		</form>
	</div>
</div>

<style lang="scss">
    form {
        button {
            &:disabled {
                opacity: 0.4;
                cursor: not-allowed;
            }
        }
    }
</style>