<script>
	import { onMount } from "svelte";
    import Chat from "./lib/pages/Chat.svelte";
	import SideBar from "./lib/pages/SideBar.svelte";
    let page;

    function onRouteChange() {
        const path = window.location.hash.slice(1)

        if (path === '/') {
            page = 'home'
        } else if (path === '/creators') {
            page = 'creators'
        } else if (path === '/settings') {
            page = 'settings'
        } else {
            window.location.hash = '/';
        }
    }
    onMount(onRouteChange);
</script>
<svelte:window on:hashchange={onRouteChange} />

<div class="h-screen w-full bg-white flex overflow-hidden">
    <SideBar  />

    <!-- Main Content -->
    <div class="w-full h-full flex flex-col">
        <main class="flex-1 overflow-y-auto p-4">
            {#if page === 'home'}
                <Chat />
            {:else if page === 'creators'}
                <div class="w-full h-40 bg-gray-400 rounded-lg flex items-center justify-center">
                    asdf
                </div>
                
            {:else if page === 'settings'}
                <h2>Settings</h2>
            {/if}
        </main>
    </div>
</div>

